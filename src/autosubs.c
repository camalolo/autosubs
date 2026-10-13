// autosubs - fully automatic SRT subtitle generation for video files
//
// Uses whisper.cpp (large-v3) to transcribe the audio track of a video
// (or every video found under a directory) and writes a perfectly synced
// .srt file next to it.
//
// Audio is decoded in-process from ffmpeg's stdout (16 kHz mono f32le),
// so no temporary files are created.
//
// Build: see CMakeLists.txt (links whisper.dll / ggml-*.dll built with CUDA).
//
// Usage: autosubs <file-or-directory> [options]   (see --help)

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS

#include <windows.h>
#include <whisper.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <ctype.h>
#include <stdint.h>

#ifndef AS_VERSION
#define AS_VERSION "1.0.0-dev"
#endif

// ---------------------------------------------------------------------------
// misc helpers
// ---------------------------------------------------------------------------

static int g_verbose = 0;

static void die(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "autosubs: error: ");
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

static void info(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    fflush(stdout);
}

static void * xmalloc(size_t n) {
    void * p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "autosubs: out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}

static void * xrealloc(void * p, size_t n) {
    void * q = realloc(p, n ? n : 1);
    if (!q) { fprintf(stderr, "autosubs: out of memory (%zu bytes)\n", n); exit(1); }
    return q;
}

static char * xstrdup(const char * s) {
    size_t n = strlen(s) + 1;
    char * p = (char *) xmalloc(n);
    memcpy(p, s, n);
    return p;
}

static wchar_t * xwcsdup(const wchar_t * s) {
    size_t n = (wcslen(s) + 1) * sizeof(wchar_t);
    wchar_t * p = (wchar_t *) xmalloc(n);
    memcpy(p, s, n);
    return p;
}

static char * utf16_to_utf8(const wchar_t * ws) {
    int n = WideCharToMultiByte(CP_UTF8, 0, ws, -1, NULL, 0, NULL, NULL);
    if (n <= 0) return xstrdup("");
    char * out = (char *) xmalloc((size_t) n);
    WideCharToMultiByte(CP_UTF8, 0, ws, -1, out, n, NULL, NULL);
    return out;
}

static wchar_t * utf8_to_utf16(const char * s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    if (n <= 0) return xwcsdup(L"");
    wchar_t * out = (wchar_t *) xmalloc((size_t) n * sizeof(wchar_t));
    MultiByteToWideChar(CP_UTF8, 0, s, -1, out, n);
    return out;
}

// ---------------------------------------------------------------------------
// console / abort handling
// ---------------------------------------------------------------------------

static volatile LONG g_abort = 0;

static BOOL WINAPI ctrl_handler(DWORD type) {
    (void) type;
    g_abort = 1;
    info("\n  interrupted - finishing up...\n");
    return TRUE;
}

// ---------------------------------------------------------------------------
// whisper logging: keep the console readable
// ---------------------------------------------------------------------------

static void whisper_log_cb(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
        fflush(stderr);
    } else if (g_verbose) {
        fprintf(stderr, "%s", text);
    }
}

// ---------------------------------------------------------------------------
// options
// ---------------------------------------------------------------------------

struct options {
    const char * model;        // utf8, NULL = auto
    const char * vad_model;    // utf8, NULL = auto
    const char * language;     // "auto" or code
    const char * ffmpeg;       // utf8, NULL = auto
    const char * output_dir;   // utf8 utf-16 wide handled separately
    const wchar_t * output_dir_w;
    const char * prompt;
    int   translate;
    int   force;
    int   dry_run;
    int   no_gpu;
    int   no_flash_attn;
    int   no_vad;
    int   beam;
    int   threads;
    int   audio_stream;
    int   chunk_secs;
    float vad_threshold;
    int   no_highpass;
    int   no_norm;
    const char * denoise_model;  // utf8 path to .rnnn model, NULL = off
    char *exts[64];
    int   n_exts;
};

static struct options g_opts = {
    .language      = "auto",
    .beam          = 5,
    .threads       = 0,     // 0 = auto
    .chunk_secs    = 180,
    .vad_threshold = 0.5f,
};

static const char * default_exts[] = {
    "mp4", "mkv", "avi", "mov", "webm", "wmv", "m4v", "mpg", "mpeg",
    "ts", "m2ts", "flv", "3gp", "ogv", "vob", "mts", "m2v", "divx", "rmvb"
};

static int ext_matches(const char * ext) {
    int i;
    for (i = 0; i < g_opts.n_exts; i++) {
        if (_stricmp(ext, g_opts.exts[i]) == 0) return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// path helpers
// ---------------------------------------------------------------------------

static BOOL path_exists_w(const wchar_t * p) {
    DWORD a = GetFileAttributesW(p);
    return a != INVALID_FILE_ATTRIBUTES;
}

static BOOL is_dir_w(const wchar_t * p) {
    DWORD a = GetFileAttributesW(p);
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

static const wchar_t * base_name_w(const wchar_t * p) {
    const wchar_t * b = p;
    const wchar_t * q = p;
    while (*q) {
        if (*q == L'\\' || *q == L'/') b = q + 1;
        q++;
    }
    return b;
}

static void strip_ext_w(wchar_t * p) {
    wchar_t * dot = NULL;
    wchar_t * q = p;
    while (*q) {
        if (*q == L'.') dot = q;
        else if (*q == L'\\' || *q == L'/') dot = NULL;
        q++;
    }
    if (dot) *dot = 0;
}

static wchar_t * get_exe_dir_w(void) {
    wchar_t buf[MAX_PATH * 2];
    DWORD n = GetModuleFileNameW(NULL, buf, sizeof(buf) / sizeof(buf[0]));
    if (n == 0 || n >= sizeof(buf) / sizeof(buf[0])) return xwcsdup(L".");
    wchar_t * slash = wcsrchr(buf, L'\\');
    if (slash) *slash = 0;
    return xwcsdup(buf);
}

static void make_dirs_w(const wchar_t * path) {
    wchar_t tmp[MAX_PATH * 4];
    wcsncpy(tmp, path, sizeof(tmp) / sizeof(tmp[0]) - 1);
    tmp[sizeof(tmp) / sizeof(tmp[0]) - 1] = 0;
    size_t len = wcslen(tmp);
    for (size_t i = 1; i < len; i++) {
        if (tmp[i] == L'\\' || tmp[i] == L'/') {
            wchar_t c = tmp[i];
            tmp[i] = 0;
            CreateDirectoryW(tmp, NULL);
            tmp[i] = c;
        }
    }
    CreateDirectoryW(tmp, NULL);
}

static const wchar_t * ext_of_w(const wchar_t * p) {
    const wchar_t * dot = NULL;
    const wchar_t * q = p;
    while (*q) {
        if (*q == L'.') dot = q;
        else if (*q == L'\\' || *q == L'/') dot = NULL;
        q++;
    }
    return dot ? dot + 1 : L"";
}

// replaces the extension of `path` with `newext` (without dot)
static wchar_t * replace_ext_w(const wchar_t * path, const wchar_t * newext) {
    size_t n = wcslen(path) + wcslen(newext) + 8;
    wchar_t * out = (wchar_t *) xmalloc(n * sizeof(wchar_t));
    wcscpy(out, path);
    strip_ext_w(out);
    wcscat(out, L".");
    wcscat(out, newext);
    return out;
}

// ---------------------------------------------------------------------------
// tool discovery (ffmpeg / ffprobe)
// ---------------------------------------------------------------------------

// Resolve a bare executable name or explicit path to a full path.
static wchar_t * resolve_executable(const wchar_t * name_or_path) {
    if (!name_or_path || !*name_or_path) return NULL;

    wchar_t full[MAX_PATH * 4];
    DWORD n = SearchPathW(NULL, name_or_path, NULL, (DWORD) (sizeof(full) / sizeof(full[0])), full, NULL);
    if (n > 0 && n < sizeof(full) / sizeof(full[0])) return xwcsdup(full);

    // explicit relative/absolute path?
    if (path_exists_w(name_or_path)) {
        if (GetFullPathNameW(name_or_path, (DWORD) (sizeof(full) / sizeof(full[0])), full, NULL) > 0) {
            return xwcsdup(full);
        }
    }
    return NULL;
}

static wchar_t * find_tool(const wchar_t * exe_name, const wchar_t * override) {
    wchar_t * r;

    if (override && *override) {
        r = resolve_executable(override);
        if (!r) {
            char * o8 = utf16_to_utf8(override);
            die("cannot find '%s'", o8);
        }
        return r;
    }

    // environment variable
    {
        wchar_t envname[64];
        if (_wcsicmp(exe_name, L"ffmpeg.exe") == 0) wcscpy(envname, L"AUTOSUBS_FFMPEG");
        else if (_wcsicmp(exe_name, L"ffprobe.exe") == 0) wcscpy(envname, L"AUTOSUBS_FFPROBE");
        else wcscpy(envname, L"AUTOSUBS_TOOL");
        wchar_t * env = _wgetenv(envname);
        if (env && *env) {
            r = resolve_executable(env);
            if (r) return r;
        }
    }

    // next to the executable
    {
        wchar_t * dir = get_exe_dir_w();
        wchar_t * cand = (wchar_t *) xmalloc((wcslen(dir) + wcslen(exe_name) + 8) * sizeof(wchar_t));
        wcscpy(cand, dir);
        wcscat(cand, L"\\");
        wcscat(cand, exe_name);
        if (path_exists_w(cand)) { free(dir); return cand; }
        free(cand);
        free(dir);
    }

    // PATH
    r = resolve_executable(exe_name);
    if (r) return r;

    // a few common install locations
    {
        static const wchar_t * common[] = {
            L"C:\\ffmpeg\\bin",
            L"C:\\Program Files\\ffmpeg\\bin",
            L"C:\\Program Files (x86)\\ffmpeg\\bin",
            L"C:\\ProgramData\\chocolatey\\bin",
        };
        for (int i = 0; i < (int) (sizeof(common) / sizeof(common[0])); i++) {
            wchar_t cand[MAX_PATH * 2];
            _snwprintf(cand, sizeof(cand) / sizeof(cand[0]), L"%s\\%s", common[i], exe_name);
            if (path_exists_w(cand)) return xwcsdup(cand);
        }
    }

    return NULL;
}

// ---------------------------------------------------------------------------
// media file collection
// ---------------------------------------------------------------------------

struct media_list {
    wchar_t ** paths;
    int        count;
    int        cap;
};

static void media_add(struct media_list * ml, const wchar_t * path) {
    if (ml->count == ml->cap) {
        ml->cap = ml->cap ? ml->cap * 2 : 64;
        ml->paths = (wchar_t **) xrealloc(ml->paths, (size_t) ml->cap * sizeof(wchar_t *));
    }
    ml->paths[ml->count++] = xwcsdup(path);
}

static int media_cmp(const void * a, const void * b) {
    return _wcsicmp(*(wchar_t * const *) a, *(wchar_t * const *) b);
}

static void scan_dir(const wchar_t * dir, struct media_list * ml, int depth) {
    if (depth > 8 || g_abort) return;

    wchar_t pattern[MAX_PATH * 4];
    _snwprintf(pattern, sizeof(pattern) / sizeof(pattern[0]), L"%s\\*", dir);

    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return;

    do {
        if (wcscmp(fd.cFileName, L".") == 0 || wcscmp(fd.cFileName, L"..") == 0) continue;

        wchar_t full[MAX_PATH * 4];
        _snwprintf(full, sizeof(full) / sizeof(full[0]), L"%s\\%s", dir, fd.cFileName);

        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            scan_dir(full, ml, depth + 1);
        } else {
            const wchar_t * ext = ext_of_w(fd.cFileName);
            char ext8[32];
            WideCharToMultiByte(CP_UTF8, 0, ext, -1, ext8, sizeof(ext8), NULL, NULL);
            if (ext_matches(ext8)) {
                media_add(ml, full);
            }
        }
    } while (FindNextFileW(h, &fd));

    FindClose(h);
}

// ---------------------------------------------------------------------------
// ffmpeg audio extraction (raw f32le 16 kHz mono straight into memory)
// ---------------------------------------------------------------------------

struct pipe_reader {
    HANDLE h;
    char * buf;
    size_t len;
    size_t cap;
};

static DWORD WINAPI stderr_reader_thread(LPVOID param) {
    struct pipe_reader * pr = (struct pipe_reader *) param;
    char tmp[4096];
    DWORD n;
    for (;;) {
        if (!ReadFile(pr->h, tmp, sizeof(tmp), &n, NULL) || n == 0) break;
        if (pr->len + n + 1 > pr->cap) {
            pr->cap = (pr->cap ? pr->cap * 2 : 8192);
            while (pr->len + n + 1 > pr->cap) pr->cap *= 2;
            pr->buf = (char *) xrealloc(pr->buf, pr->cap);
        }
        memcpy(pr->buf + pr->len, tmp, n);
        pr->len += n;
        pr->buf[pr->len] = 0;
    }
    return 0;
}

static wchar_t * build_ffmpeg_cmd(const wchar_t * ffmpeg, const wchar_t * input, int audio_stream) {
    // build the audio filter chain
    wchar_t afilter[512] = L"";
    if (!g_opts.no_highpass) wcscpy(afilter, L"highpass=f=50:poles=2");
    if (g_opts.denoise_model) {
        // RNNoise speech enhancement (opt-in; can harm music-heavy audio)
        wchar_t * dm = utf8_to_utf16(g_opts.denoise_model);
        wchar_t tmp[400];
        _snwprintf(tmp, 400, L"%sarnndn=m=\"%s\"", afilter[0] ? afilter : L"", dm);
        wcscpy(afilter, tmp);
        free(dm);
    }

    // worst case size
    size_t need = wcslen(ffmpeg) + wcslen(input) + wcslen(afilter) + 256;
    wchar_t * cmd = (wchar_t *) xmalloc(need * sizeof(wchar_t));
    if (afilter[0]) {
        _snwprintf(cmd, need, L"\"%s\" -hide_banner -nostdin -v error -i \"%s\" "
                              L"-vn -sn -dn -map 0:a:%d? -ac 1 -ar 16000 -c:a pcm_f32le -af \"%s\" -f f32le -",
                   ffmpeg, input, audio_stream, afilter);
    } else {
        _snwprintf(cmd, need, L"\"%s\" -hide_banner -nostdin -v error -i \"%s\" "
                              L"-vn -sn -dn -map 0:a:%d? -ac 1 -ar 16000 -c:a pcm_f32le -f f32le -",
                   ffmpeg, input, audio_stream);
    }
    return cmd;
}

// static RMS gain: center the track around a healthy speech level without
// touching dynamics (no pumping artifacts). whisper's log-mel is relative,
// but this helps very quiet sources and keeps levels sane.
static void normalize_rms(float * samples, size_t n) {
    if (n == 0) return;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += (double) samples[i] * samples[i];
    double rms = sqrt(sum / (double) n);
    if (rms < 1e-6) return;                       // digital silence
    const double target = 0.0708;                 // ~ -23 dBFS
    double gain = target / rms;
    if (gain > 10.0) gain = 10.0;                 // cap at +20 dB
    if (gain < 1.0 || fabs(gain - 1.0) < 0.05) return; // only boost, and only if it matters
    for (size_t i = 0; i < n; i++) samples[i] = (float) (samples[i] * gain);
}

// Extract audio -> samples (owned by caller, free()).
// Returns 0 on success. `err` receives a diagnostic message on failure.
static int extract_audio(const wchar_t * ffmpeg, const wchar_t * input, int audio_stream,
                         float ** out_samples, size_t * out_count,
                         double * out_secs, char * err, size_t errcap) {
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = NULL;
    sa.bInheritHandle = TRUE;

    HANDLE out_rd = NULL, out_wr = NULL, err_rd = NULL, err_wr = NULL;

    if (!CreatePipe(&out_rd, &out_wr, &sa, 0)) { snprintf(err, errcap, "CreatePipe failed"); return -1; }
    if (!CreatePipe(&err_rd, &err_wr, &sa, 0)) { snprintf(err, errcap, "CreatePipe failed"); CloseHandle(out_rd); CloseHandle(out_wr); return -1; }
    SetHandleInformation(out_rd, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(err_rd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = NULL;
    si.hStdOutput = out_wr;
    si.hStdError  = err_wr;

    wchar_t * cmd = build_ffmpeg_cmd(ffmpeg, input, audio_stream);

    DWORD t0 = GetTickCount();
    BOOL ok = CreateProcessW(NULL, cmd, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    free(cmd);

    if (!ok) {
        snprintf(err, errcap, "failed to start ffmpeg (error %lu)", GetLastError());
        CloseHandle(out_rd); CloseHandle(out_wr); CloseHandle(err_rd); CloseHandle(err_wr);
        return -1;
    }

    CloseHandle(out_wr);
    CloseHandle(err_wr);

    struct pipe_reader pr;
    pr.h = err_rd; pr.buf = NULL; pr.len = 0; pr.cap = 0;
    HANDLE th = CreateThread(NULL, 0, stderr_reader_thread, &pr, 0, NULL);

    size_t cap = 16u << 20; // 16 MB initial (4 min of audio)
    float * samples = (float *) xmalloc(cap);
    size_t len = 0;

    char * byte_buf = (char *) samples;
    for (;;) {
        if (len + (1u << 20) > cap) {
            cap *= 2;
            samples = (float *) xrealloc(samples, cap);
            byte_buf = (char *) samples;
        }
        DWORD got = 0;
        if (!ReadFile(out_rd, byte_buf + len, (DWORD) (1u << 20), &got, NULL) || got == 0) break;
        len += got;
        if (g_abort) break;
    }

    CloseHandle(out_rd);

    DWORD wait = WaitForSingleObject(pi.hProcess, 120000);
    DWORD code = (DWORD) -1;
    if (wait == WAIT_OBJECT_0) GetExitCodeProcess(pi.hProcess, &code);
    else if (wait == WAIT_TIMEOUT) { TerminateProcess(pi.hProcess, 1); snprintf(err, errcap, "ffmpeg timed out"); }

    if (th) { WaitForSingleObject(th, 5000); CloseHandle(th); }
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(err_rd);

    if (g_abort) {
        free(pr.buf); free(samples);
        snprintf(err, errcap, "aborted");
        return -1;
    }

    if (wait == WAIT_TIMEOUT || code != 0 || len == 0) {
        // last non-empty stderr line as diagnosis
        char * msg = pr.buf ? pr.buf : (char *) "";
        size_t mlen = strlen(msg);
        while (mlen > 0 && (msg[mlen - 1] == '\n' || msg[mlen - 1] == '\r' || msg[mlen - 1] == ' ')) msg[--mlen] = 0;
        if (len == 0 && mlen == 0) snprintf(err, errcap, "ffmpeg produced no audio (exit code %lu)", code);
        else snprintf(err, errcap, "%.*s", (int) (errcap - 1), msg);
        free(pr.buf);
        free(samples);
        return -1;
    }

    free(pr.buf);

    *out_samples = samples;
    *out_count   = len / sizeof(float);
    *out_secs    = (GetTickCount() - t0) / 1000.0;
    return 0;
}

// probe duration (seconds) via ffprobe; returns 0.0 on failure
static double probe_duration(const wchar_t * ffprobe, const wchar_t * input) {
    if (!ffprobe) return 0.0;

    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = NULL;
    sa.bInheritHandle = TRUE;

    HANDLE rd = NULL, wr = NULL;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return 0.0;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = NULL;
    si.hStdOutput = wr;
    si.hStdError  = NULL;

    size_t need = wcslen(ffprobe) + wcslen(input) + 256;
    wchar_t * cmd = (wchar_t *) xmalloc(need * sizeof(wchar_t));
    _snwprintf(cmd, need, L"\"%s\" -hide_banner -v error -show_entries format=duration "
                          L"-of default=nw=1:nk=1 \"%s\"", ffprobe, input);

    BOOL ok = CreateProcessW(NULL, cmd, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    free(cmd);
    CloseHandle(wr);
    if (!ok) { CloseHandle(rd); return 0.0; }

    char buf[128];
    DWORD got = 0;
    ReadFile(rd, buf, sizeof(buf) - 1, &got, NULL);
    buf[got < sizeof(buf) ? got : sizeof(buf) - 1] = 0;
    CloseHandle(rd);

    WaitForSingleObject(pi.hProcess, 15000);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return atof(buf);
}

// ---------------------------------------------------------------------------
// whisper callbacks
// ---------------------------------------------------------------------------

struct progress_state {
    int    last_pct;
    DWORD  last_tick;
    int64_t base_ms;   // start of current chunk
    int64_t dur_ms;    // duration of current chunk
    int64_t total_ms;  // total audio duration
};

static void progress_cb(struct whisper_context * ctx, struct whisper_state * state, int progress, void * user_data) {
    (void) ctx; (void) state;
    struct progress_state * ps = (struct progress_state *) user_data;
    int64_t done = ps->base_ms + (int64_t) progress * ps->dur_ms / 100;
    int pct = ps->total_ms > 0 ? (int) (done * 100 / ps->total_ms) : progress;
    if (pct > 100) pct = 100;
    DWORD now = GetTickCount();
    if (pct != ps->last_pct && (pct == 100 || now - ps->last_tick > 100)) {
        int filled = pct / 5;
        char bar[24];
        for (int i = 0; i < 20; i++) bar[i] = i < filled ? '#' : '.';
        bar[20] = 0;
        info("\r    [%s] %3d%%", bar, pct);
        ps->last_pct = pct;
        ps->last_tick = now;
    }
}

static void new_segment_cb(struct whisper_context * ctx, struct whisper_state * state, int n_new, void * user_data) {
    (void) state; (void) user_data;
    if (!g_verbose) return;
    int n = whisper_full_n_segments(ctx);
    for (int i = n - n_new; i < n; i++) {
        int64_t t0 = whisper_full_get_segment_t0(ctx, i) * 10;
        int64_t t1 = whisper_full_get_segment_t1(ctx, i) * 10;
        const char * text = whisper_full_get_segment_text(ctx, i);
        info("\n    [%02d:%02d:%02d.%03d -> %02d:%02d:%02d.%03d] %s",
             (int) (t0 / 3600000), (int) (t0 / 60000 % 60), (int) (t0 / 1000 % 60), (int) (t0 % 1000),
             (int) (t1 / 3600000), (int) (t1 / 60000 % 60), (int) (t1 / 1000 % 60), (int) (t1 % 1000),
             text);
    }
}

static bool abort_cb(void * user_data) {
    (void) user_data;
    return g_abort != 0;
}

// ---------------------------------------------------------------------------
// cue collection + SRT output
// ---------------------------------------------------------------------------

static char * clean_text(const char * in);

struct cue {
    int64_t t0, t1; // ms
    char *  text;
};

struct cue_list {
    struct cue * items;
    int count, cap;
};

static void cue_add(struct cue_list * cl, int64_t t0, int64_t t1, const char * text) {
    char * cleaned = clean_text(text);
    if (cleaned[0] == 0) { free(cleaned); return; }
    if (cl->count == cl->cap) {
        cl->cap = cl->cap ? cl->cap * 2 : 256;
        cl->items = (struct cue *) xrealloc(cl->items, (size_t) cl->cap * sizeof(struct cue));
    }
    cl->items[cl->count].t0 = t0;
    cl->items[cl->count].t1 = t1;
    cl->items[cl->count].text = cleaned;
    cl->count++;
}

static void cue_list_free(struct cue_list * cl) {
    for (int i = 0; i < cl->count; i++) free(cl->items[i].text);
    free(cl->items);
    cl->items = NULL; cl->count = cl->cap = 0;
}

// collect whisper segments from the last whisper_full() call into `cues`
static void collect_cues(struct whisper_context * ctx, int64_t base_ms, struct cue_list * cues) {
    int n_seg = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_seg; i++) {
        int64_t t0ms = whisper_full_get_segment_t0(ctx, i) * 10 + base_ms;
        int64_t t1ms = whisper_full_get_segment_t1(ctx, i) * 10 + base_ms;
        cue_add(cues, t0ms, t1ms, whisper_full_get_segment_text(ctx, i));
    }
}

// transcribe [s0, s1) samples; appends cues (absolute ms)
static int transcribe_range(struct whisper_context * ctx, struct whisper_full_params wparams,
                            const float * samples, int64_t s0, int64_t s1,
                            struct progress_state * ps, int64_t base_ms,
                            struct cue_list * cues) {
    ps->base_ms = base_ms;
    ps->dur_ms  = (s1 - s0) * 1000 / WHISPER_SAMPLE_RATE;
    int ret = whisper_full(ctx, wparams, samples + s0, (int) (s1 - s0));
    if (ret != 0) return ret;
    collect_cues(ctx, base_ms, cues);
    return 0;
}

// post-filter: remove hallucination loops and unrealistic cue durations
static void cue_postprocess(struct cue_list * cl) {
    int w = 0;
    for (int i = 0; i < cl->count; i++) {
        // collapse hallucination loops: 3+ consecutive identical texts -> keep first.
        // (real echoes are at most 2 repeats; decoder loops regurgitate the same
        //  line dozens of times across window boundaries)
        int j = i;
        while (j + 1 < cl->count && strcmp(cl->items[j + 1].text, cl->items[i].text) == 0) j++;
        int run = j - i + 1;
        if (run >= 3) {
            cl->items[w++] = cl->items[i];
            for (int k = i + 1; k <= j; k++) free(cl->items[k].text);
            i = j;
            continue;
        }
        // immediate identical repeat (<200 ms gap) is always a loop artifact
        if (w > 0) {
            struct cue * prev = &cl->items[w - 1];
            if (strcmp(prev->text, cl->items[i].text) == 0 && cl->items[i].t0 - prev->t1 < 200) {
                free(cl->items[i].text);
                continue;
            }
        }
        cl->items[w++] = cl->items[i];
    }
    cl->count = w;

    // collapse cyclic repetition (a sequence of K cues that repeats): keep the
    // first cycle, stretched across the full span of the loop
    {
        struct cue * out = (struct cue *) xmalloc((size_t) (cl->count + 1) * sizeof(struct cue));
        int w2 = 0;
        int i = 0;
        while (i < cl->count) {
            int K = 0, cyc = 0;
            for (int k = 1; k <= 12 && i + 2 * k <= cl->count; k++) {
                int ok = 1;
                for (int q = 0; q < k; q++) {
                    if (strcmp(cl->items[i + q].text, cl->items[i + k + q].text) != 0) { ok = 0; break; }
                }
                if (!ok) continue;
                int c = 2;
                while (i + (c + 1) * k <= cl->count) {
                    int ok2 = 1;
                    for (int q = 0; q < k; q++) {
                        if (strcmp(cl->items[i + c * k + q].text, cl->items[i + q].text) != 0) { ok2 = 0; break; }
                    }
                    if (!ok2) break;
                    c++;
                }
                if ((k == 1 && c >= 3) || (k >= 2 && c >= 2)) { K = k; cyc = c; break; }
            }
            if (K == 0) { out[w2++] = cl->items[i++]; continue; }

            int end = i + K * cyc;
            if (g_verbose) {
                info("    [collapse] cycle K=%d x%d @%.1fs -> %.1fs (keeping %d)\n",
                     K, cyc, cl->items[i].t0 / 1000.0, cl->items[end - 1].t1 / 1000.0, K);
            }
            int64_t span0 = cl->items[i].t0;
            int64_t span1 = cl->items[end - 1].t1;
            double scale = 1.0;
            if (K >= 2) {
                int64_t c0 = cl->items[i].t0;
                int64_t c1 = cl->items[i + K - 1].t1;
                if (c1 > c0 && span1 > span0) {
                    scale = (double) (span1 - span0) / (double) (c1 - c0);
                    if (scale < 1.0 || scale > 4.0) scale = 1.0;
                }
            }
            for (int q = 0; q < K; q++) {
                struct cue * c = &cl->items[i + q];
                if (scale > 1.0) {
                    c->t0 = span0 + (int64_t) ((double) (c->t0 - span0) * scale);
                    c->t1 = span0 + (int64_t) ((double) (c->t1 - span0) * scale);
                }
                out[w2++] = *c;
            }
            for (int q = K; q < K * cyc; q++) free(cl->items[i + q].text);
            i = end;
        }
        cl->count = w2;
        free(cl->items);
        cl->items = out;
        cl->cap = w2;
    }

    // resolve overlaps before judging cue sanity. only small overlaps are
    // resolved here; large ones (degenerate spans) are left for write_srt so
    // a single bad cue can never cascade-delete the rest.
    for (int i = 1; i < cl->count; i++) {
        if (cl->items[i].t0 < cl->items[i - 1].t1 && cl->items[i - 1].t1 - cl->items[i].t0 <= 5000) {
            cl->items[i].t0 = cl->items[i - 1].t1;
        }
        if (cl->items[i].t1 <= cl->items[i].t0) cl->items[i].t1 = cl->items[i].t0 + 10;
    }

    for (int i = 0; i < cl->count; i++) {
        int64_t cap = (int64_t) strlen(cl->items[i].text) * 120 + 3000; // ~8 chars/s + 3 s floor
        if (cap > 20000) cap = 20000;
        if (cl->items[i].t1 - cl->items[i].t0 > cap) cl->items[i].t1 = cl->items[i].t0 + cap;
    }

    // drop junk cues: punctuation-only text, or physically impossible speech
    // rates (hallucination leaks: several words crammed into a 100 ms cue)
    {
        int w3 = 0, n_punct = 0, n_rate = 0;
        for (int i = 0; i < cl->count; i++) {
            const char * t = cl->items[i].text;
            int alnum = 0;
            for (const char * p = t; *p; p++) {
                if (isalnum((unsigned char) *p) || (unsigned char) *p >= 0x80) { alnum = 1; break; }
            }
            if (!alnum) { free(cl->items[i].text); n_punct++; continue; }

            double dur = (double) (cl->items[i].t1 - cl->items[i].t0) / 1000.0;
            double rate = (double) strlen(t) / (dur > 0.05 ? dur : 0.05);
            if (rate > 22.0) {
                if (g_verbose) {
                    info("    [drop-rate] @%.1fs %.0f c/s: %.40s\n",
                         cl->items[i].t0 / 1000.0, rate, t);
                }
                free(cl->items[i].text);
                n_rate++;
                continue;
            }
            cl->items[w3++] = cl->items[i];
        }
        cl->count = w3;
        if (g_verbose && (n_punct + n_rate > 0)) {
            info("    postprocess: dropped %d punct-only, %d impossible-rate\n", n_punct, n_rate);
        }
    }
}

// find the first cyclic repetition (decoder loop); returns indices [si, ei)
static int find_cycle(const struct cue_list * cl, int from, int * si_out, int * ei_out) {
    int n = cl->count;
    for (int i = from; i < n; i++) {
        for (int K = 1; K <= 12 && i + 2 * K <= n; K++) {
            int ok = 1;
            for (int q = 0; q < K; q++) {
                if (strcmp(cl->items[i + q].text, cl->items[i + K + q].text) != 0) { ok = 0; break; }
            }
            if (!ok) continue;
            int c = 2;
            while (i + (c + 1) * K <= n) {
                int ok2 = 1;
                for (int q = 0; q < K; q++) {
                    if (strcmp(cl->items[i + c * K + q].text, cl->items[i + q].text) != 0) { ok2 = 0; break; }
                }
                if (!ok2) break;
                c++;
            }
            if ((K == 1 && c >= 3) || (K >= 2 && c >= 2)) {
                *si_out = i;
                *ei_out = i + K * c;
                return 1;
            }
        }
    }
    return 0;
}

// replace the cues intersecting [ms0, ms1) with the cues in `rep` (ownership moves)
static void cue_splice_replace(struct cue_list * cl, int64_t ms0, int64_t ms1, struct cue_list * rep) {
    struct cue * out = (struct cue *) xmalloc((size_t) (cl->count + rep->count + 1) * sizeof(struct cue));
    int w = 0, inserted = 0, n_removed = 0, first_removed = -1, last_removed = -1;
    for (int i = 0; i < cl->count; i++) {
        struct cue * c = &cl->items[i];
        if ((c->t1 > ms0 && c->t0 < ms1)) {
            if (g_verbose) {
                info("    [splice] removing cue @%.1fs->%.1fs: %.30s\n", c->t0 / 1000.0, c->t1 / 1000.0, c->text);
            }
            if (first_removed < 0) first_removed = i;
            last_removed = i;
            n_removed++;
            free(c->text);
            continue;
        }
        // insert the replacement right after the last removed cue (i.e. at the
        // repaired position), NOT before the first surviving cue
        if (!inserted && last_removed >= 0) {
            for (int j = 0; j < rep->count; j++) out[w++] = rep->items[j];
            inserted = 1;
        }
        out[w++] = *c;
    }
    if (!inserted) {
        for (int j = 0; j < rep->count; j++) out[w++] = rep->items[j];
    }
    if (g_verbose) {
        info("    [splice] span %.1fs->%.1fs: removed %d cues (idx %d..%d), inserted %d\n",
             ms0 / 1000.0, ms1 / 1000.0, n_removed, first_removed, last_removed, rep->count);
    }
    free(cl->items);
    cl->items = out;
    cl->count = w;
    cl->cap = w;
    rep->items = NULL; rep->count = rep->cap = 0;
}

static void format_ts(int64_t ms, char * out) {
    if (ms < 0) ms = 0;
    int h = (int) (ms / 3600000); ms %= 3600000;
    int m = (int) (ms / 60000);   ms %= 60000;
    int s = (int) (ms / 1000);    ms %= 1000;
    sprintf(out, "%02d:%02d:%02d,%03d", h, m, s, (int) ms);
}

// trim + collapse whitespace/newlines from a whisper segment text
static char * clean_text(const char * in) {
    size_t n = strlen(in);
    char * out = (char *) xmalloc(n + 1);
    size_t o = 0;
    int pending_space = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char) in[i];
        if (c == '\r' || c == '\n' || c == '\t') c = ' ';
        if (c == ' ') {
            if (o > 0) pending_space = 1;
            continue;
        }
        if (pending_space) { out[o++] = ' '; pending_space = 0; }
        out[o++] = (char) c;
    }
    out[o] = 0;
    return out;
}

static int write_srt(const wchar_t * srt_path, const struct cue_list * cl) {
    if (cl->count <= 0) return -1;

    FILE * f = _wfopen(srt_path, L"wb");
    if (!f) return -1;

    int64_t prev_t1 = -1;

    for (int i = 0; i < cl->count; i++) {
        int64_t t0 = cl->items[i].t0;
        int64_t t1 = cl->items[i].t1;
        const char * text = cl->items[i].text;

        // repair timings
        if (t1 <= t0) t1 = t0 + 100;
        if (prev_t1 >= 0 && t0 < prev_t1) t0 = prev_t1;       // no overlaps
        if (t1 <= t0) t1 = t0 + 100;

        char ts0[32], ts1[32];
        format_ts(t0, ts0);
        format_ts(t1, ts1);

        fprintf(f, "%d\n%s --> %s\n%s\n\n", i + 1, ts0, ts1, text);
        prev_t1 = t1;
    }

    fclose(f);
    return cl->count;
}

// ---------------------------------------------------------------------------
// model discovery
// ---------------------------------------------------------------------------

// returns utf8 path (owned by caller) or NULL.
// `explicit_path` is utf8 (CLI/env origin); `env_name`/`file_name` are wide.
static char * find_model_file(const char * explicit_path, const wchar_t * env_name, const wchar_t * file_name) {
    wchar_t cand[MAX_PATH * 4];
    wchar_t * exe_dir = get_exe_dir_w();

    if (explicit_path && *explicit_path) {
        wchar_t * w = utf8_to_utf16(explicit_path);
        int ok = path_exists_w(w);
        free(w);
        if (ok) { free(exe_dir); return xstrdup(explicit_path); }
        free(exe_dir);
        return NULL;
    }

    const wchar_t * dirs[3];
    int n_dirs = 0;
    dirs[n_dirs++] = exe_dir;
    {
        // exe_dir\..
        static wchar_t parent[MAX_PATH * 2];
        _snwprintf(parent, sizeof(parent) / sizeof(parent[0]), L"%s\\..", exe_dir);
        dirs[n_dirs++] = parent;
    }
    dirs[n_dirs++] = L".";

    for (int i = 0; i < n_dirs; i++) {
        _snwprintf(cand, sizeof(cand) / sizeof(cand[0]), L"%s\\models\\%s", dirs[i], file_name);
        if (path_exists_w(cand)) { free(exe_dir); return utf16_to_utf8(cand); }
    }
    for (int i = 0; i < n_dirs; i++) {
        _snwprintf(cand, sizeof(cand) / sizeof(cand[0]), L"%s\\%s", dirs[i], file_name);
        if (path_exists_w(cand)) { free(exe_dir); return utf16_to_utf8(cand); }
    }

    wchar_t * env = _wgetenv(env_name);
    if (env && *env) {
        int ok = path_exists_w(env);
        if (ok) { free(exe_dir); return utf16_to_utf8(env); }
    }

    free(exe_dir);
    return NULL;
}

// ---------------------------------------------------------------------------
// usage
// ---------------------------------------------------------------------------

static void usage(void) {
    printf(
        "autosubs " AS_VERSION " - automatic, GPU-accelerated SRT generation (whisper.cpp large-v3)\n"
        "\n"
        "Usage:\n"
        "  autosubs <file-or-directory> [more...] [options]\n"
        "\n"
        "For every video found, the audio track is transcribed and a .srt file is\n"
        "written next to it (existing .srt files are skipped unless --force).\n"
        "\n"
        "Options:\n"
        "  -m, --model <path>       ggml model file (default: models/ggml-large-v3.bin)\n"
        "      --vad-model <path>   silero VAD model (default: models/ggml-silero-v6.2.0.bin)\n"
        "  -l, --language <code>    spoken language code, or 'auto' (default: auto)\n"
        "      --translate          translate to English instead of transcribing\n"
        "  -o, --output-dir <dir>   write .srt files into <dir> (default: beside video)\n"
        "      --audio-stream <n>   audio stream index to use (default: 0)\n"
        "      --beam <n>           beam size, 1 = greedy (default: 5)\n"
        "      --chunk <sec>        transcribe in <sec> chunks with fresh context,\n"
        "                           0 = single pass (default: 180)\n"
        "  -t, --threads <n>        CPU threads (default: half the logical cores)\n"
        "      --ffmpeg <path>      ffmpeg executable to use\n"
        "      --prompt <text>      initial prompt (names, jargon, style)\n"
        "      --denoise <model>    RNNoise speech enhancement (.rnnn model), opt-in\n"
        "      --no-highpass        disable the 50 Hz rumble filter\n"
        "      --no-norm            disable static loudness normalization\n"
        "      --no-vad             disable voice-activity detection\n"
        "      --vad-threshold <f>  VAD speech threshold 0.0-1.0 (default: 0.5)\n"
        "      --no-gpu             run on CPU\n"
        "      --no-flash-attn      disable flash attention\n"
        "      --ext <list>         comma separated media extensions to scan\n"
        "  -f, --force              overwrite existing .srt files\n"
        "  -n, --dry-run            only list what would be done\n"
        "  -v, --verbose            print each segment while transcribing\n"
        "  -h, --help               show this help\n"
        "\n"
        "Examples:\n"
        "  autosubs \"E:\\Downloads\\Planetes\\Season 01\"\n"
        "  autosubs movie.mkv --language ja\n"
        "  autosubs series/ --force --beam 5\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int wmain(int argc, wchar_t ** argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    for (int i = 0; i < (int) (sizeof(default_exts) / sizeof(default_exts[0])); i++) {
        g_opts.exts[g_opts.n_exts++] = xstrdup(default_exts[i]);
    }

    struct media_list inputs;
    inputs.paths = NULL; inputs.count = 0; inputs.cap = 0;

    // ---- parse arguments ----
    for (int i = 1; i < argc; i++) {
        char * a = utf16_to_utf8(argv[i]);

        if (a[0] == '-') {
            if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) { usage(); return 0; }
            else if (strcmp(a, "-m") == 0 || strcmp(a, "--model") == 0) { if (++i >= argc) { die("missing value for %s", a); return 1; } free((void*)g_opts.model); g_opts.model = xstrdup(utf16_to_utf8(argv[i])); }
            else if (strcmp(a, "--vad-model") == 0) { if (++i >= argc) { die("missing value for %s", a); return 1; } g_opts.vad_model = xstrdup(utf16_to_utf8(argv[i])); }
            else if (strcmp(a, "-l") == 0 || strcmp(a, "--language") == 0) { if (++i >= argc) { die("missing value for %s", a); return 1; } g_opts.language = xstrdup(utf16_to_utf8(argv[i])); }
            else if (strcmp(a, "--translate") == 0) g_opts.translate = 1;
            else if (strcmp(a, "-o") == 0 || strcmp(a, "--output-dir") == 0) { if (++i >= argc) { die("missing value for %s", a); return 1; } g_opts.output_dir = xstrdup(utf16_to_utf8(argv[i])); g_opts.output_dir_w = xwcsdup(argv[i]); }
            else if (strcmp(a, "--audio-stream") == 0) { if (++i >= argc) { die("missing value for %s", a); return 1; } g_opts.audio_stream = atoi(utf16_to_utf8(argv[i])); }
            else if (strcmp(a, "--beam") == 0) { if (++i >= argc) { die("missing value for %s", a); return 1; } g_opts.beam = atoi(utf16_to_utf8(argv[i])); if (g_opts.beam < 1) g_opts.beam = 1; }
            else if (strcmp(a, "--chunk") == 0) { if (++i >= argc) { die("missing value for %s", a); return 1; } g_opts.chunk_secs = atoi(utf16_to_utf8(argv[i])); if (g_opts.chunk_secs < 0) g_opts.chunk_secs = 0; }
            else if (strcmp(a, "--vad-threshold") == 0) { if (++i >= argc) { die("missing value for %s", a); return 1; } g_opts.vad_threshold = (float) atof(utf16_to_utf8(argv[i])); }
            else if (strcmp(a, "-t") == 0 || strcmp(a, "--threads") == 0) { if (++i >= argc) { die("missing value for %s", a); return 1; } g_opts.threads = atoi(utf16_to_utf8(argv[i])); }
            else if (strcmp(a, "--ffmpeg") == 0) { if (++i >= argc) { die("missing value for %s", a); return 1; } g_opts.ffmpeg = xstrdup(utf16_to_utf8(argv[i])); }
            else if (strcmp(a, "--prompt") == 0) { if (++i >= argc) { die("missing value for %s", a); return 1; } g_opts.prompt = xstrdup(utf16_to_utf8(argv[i])); }
            else if (strcmp(a, "--denoise") == 0) {
                if (++i >= argc) { die("missing value for %s", a); return 1; }
                g_opts.denoise_model = xstrdup(utf16_to_utf8(argv[i]));
                if (!path_exists_w(utf8_to_utf16(g_opts.denoise_model))) {
                    die("denoise model not found: %s\n"
                        "  RNNoise models (.rnnn) e.g. from github.com/GregorR/rnnoise-models", g_opts.denoise_model);
                }
            }
            else if (strcmp(a, "--no-highpass") == 0) g_opts.no_highpass = 1;
            else if (strcmp(a, "--no-norm") == 0) g_opts.no_norm = 1;
            else if (strcmp(a, "--no-vad") == 0) g_opts.no_vad = 1;
            else if (strcmp(a, "--no-gpu") == 0) g_opts.no_gpu = 1;
            else if (strcmp(a, "--no-flash-attn") == 0) g_opts.no_flash_attn = 1;
            else if (strcmp(a, "-f") == 0 || strcmp(a, "--force") == 0) g_opts.force = 1;
            else if (strcmp(a, "-n") == 0 || strcmp(a, "--dry-run") == 0) g_opts.dry_run = 1;
            else if (strcmp(a, "-v") == 0 || strcmp(a, "--verbose") == 0) g_verbose = 1;
            else if (strcmp(a, "--ext") == 0) {
                if (++i >= argc) { die("missing value for %s", a); return 1; }
                char * list = utf16_to_utf8(argv[i]);
                g_opts.n_exts = 0;
                for (char * tok = strtok(list, ",; "); tok; tok = strtok(NULL, ",; ")) {
                    if (g_opts.n_exts < 64) g_opts.exts[g_opts.n_exts++] = xstrdup(tok);
                }
                free(list);
            }
            else { die("unknown option '%s' (see --help)", a); return 1; }
        } else {
            inputs.paths = (wchar_t **) xrealloc(inputs.paths, (size_t) (inputs.count + 1) * sizeof(wchar_t *));
            inputs.paths[inputs.count++] = xwcsdup(argv[i]);
        }

        free(a);
    }

    if (inputs.count == 0) { usage(); return 1; }
    if (g_opts.dry_run) g_verbose = 0;

    // ---- locate models ----
    char * model8 = find_model_file(g_opts.model, L"AUTOSUBS_MODEL", L"ggml-large-v3.bin");
    if (!model8) {
        die("whisper model not found.\n"
            "  Put ggml-large-v3.bin into a 'models' folder next to autosubs.exe, or pass --model <path>.");
    }

    char * vad8 = NULL;
    if (!g_opts.no_vad) {
        vad8 = find_model_file(g_opts.vad_model, L"AUTOSUBS_VAD_MODEL", L"ggml-silero-v6.2.0.bin");
        if (!vad8) vad8 = find_model_file(NULL, L"AUTOSUBS_VAD_MODEL", L"ggml-silero-v5.1.2.bin");
        if (!vad8) {
            info("note: VAD model not found, continuing without voice-activity detection\n"
                 "      (fetch models/ggml-silero-v6.2.0.bin for faster, cleaner results)\n");
        }
    }

    // ---- locate ffmpeg ----
    wchar_t * ffmpeg = find_tool(L"ffmpeg.exe", g_opts.ffmpeg ? utf8_to_utf16(g_opts.ffmpeg) : NULL);
    if (!ffmpeg) {
        die("ffmpeg not found.\n"
            "  Install ffmpeg and make sure it is in PATH, place ffmpeg.exe next to autosubs.exe,\n"
            "  or pass --ffmpeg <path>.");
        return 1;
    }
    wchar_t * ffprobe = find_tool(L"ffprobe.exe", NULL);

    // ---- collect media files ----
    struct media_list media;
    media.paths = NULL; media.count = 0; media.cap = 0;

    for (int i = 0; i < inputs.count; i++) {
        const wchar_t * p = inputs.paths[i];
        if (is_dir_w(p)) {
            scan_dir(p, &media, 0);
        } else if (path_exists_w(p)) {
            media_add(&media, p);
        } else {
            char * p8 = utf16_to_utf8(p);
            info("warning: '%s' does not exist - skipped\n", p8);
            free(p8);
        }
    }

    if (media.count > 1) qsort(media.paths, (size_t) media.count, sizeof(wchar_t *), media_cmp);

    // ---- decide what to do ----
    int n_todo = 0, n_skip = 0;
    for (int i = 0; i < media.count; i++) {
        wchar_t * srt = replace_ext_w(media.paths[i], L"srt");
        if (!g_opts.force && path_exists_w(srt)) { n_skip++; free(srt); continue; }
        free(srt);
        n_todo++;
    }

    info("autosubs " AS_VERSION " - whisper large-v3%s\n", g_opts.no_gpu ? " (CPU)" : " (GPU)");
    info("model:  %s\n", model8);
    info("media:  %d file(s) found, %d to process, %d already have subtitles\n",
         media.count, n_todo, n_skip);

    if (n_todo == 0) {
        info("nothing to do.\n");
        return 0;
    }

    if (g_opts.dry_run) {
        for (int i = 0; i < media.count; i++) {
            wchar_t * srt = replace_ext_w(media.paths[i], L"srt");
            char * p8 = utf16_to_utf8(base_name_w(media.paths[i]));
            if (g_opts.force || !path_exists_w(srt)) info("  [process] %s\n", p8);
            else info("  [skip]    %s\n", p8);
            free(p8); free(srt);
        }
        return 0;
    }

    // ---- init whisper ----
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = g_opts.no_gpu ? 0 : 1;
    cparams.flash_attn = (!g_opts.no_gpu && !g_opts.no_flash_attn) ? 1 : 0;
    cparams.gpu_device = 0;

    whisper_log_set(whisper_log_cb, NULL);

    info("loading model (this takes a few seconds)...\n");
    DWORD t_load = GetTickCount();
    struct whisper_context * ctx = whisper_init_from_file_with_params(model8, cparams);
    if (!ctx) {
        die("failed to load whisper model '%s'", model8);
        return 1;
    }
    info("model loaded in %.1fs\n\n", (GetTickCount() - t_load) / 1000.0);

    if (g_opts.threads <= 0) {
        DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
        int t = (int) n / 2;
        if (t < 4) t = 4;
        if (t > 8) t = 8;
        g_opts.threads = t;
    }

    // ---- init VAD once (reused across files) ----
    struct whisper_vad_context * vad_ctx = NULL;
    struct whisper_vad_params vadp = whisper_vad_default_params();
    vadp.threshold = g_opts.vad_threshold;
    if (vad8 && g_opts.chunk_secs > 0) {
        struct whisper_vad_context_params vp = whisper_vad_default_context_params();
        vp.n_threads = g_opts.threads;
        vp.use_gpu   = false; // CUDA path hits unsupported ops for the VAD LSTM; CPU is plenty fast
        vp.gpu_device = 0;
        vad_ctx = whisper_vad_init_from_file_with_params(vad8, vp);
        if (!vad_ctx) {
            info("note: failed to load VAD model - using single-pass mode\n");
        }
    }

    // ---- process files ----
    DWORD t_start = GetTickCount();
    int n_done = 0, n_failed = 0;
    double total_audio_secs = 0.0;
    double total_proc_secs  = 0.0;

    int file_index = 0;
    for (int i = 0; i < media.count && !g_abort; i++) {
        const wchar_t * video = media.paths[i];
        wchar_t * srt_path = replace_ext_w(video, L"srt");
        char * video8 = utf16_to_utf8(video);
        char * name8  = utf16_to_utf8(base_name_w(video));

        if (!g_opts.force && path_exists_w(srt_path)) { free(video8); free(name8); free(srt_path); continue; }
        file_index++;

        info("[%d/%d] %s\n", file_index, n_todo, name8);

        // output path (optionally redirected)
        wchar_t * out_path = NULL;
        if (g_opts.output_dir_w) {
            size_t need = wcslen(g_opts.output_dir_w) + wcslen(base_name_w(video)) + 8;
            wchar_t * tmp = (wchar_t *) xmalloc(need * sizeof(wchar_t));
            wcscpy(tmp, g_opts.output_dir_w);
            wcscat(tmp, L"\\");
            wcscat(tmp, base_name_w(video));
            strip_ext_w(tmp);
            wcscat(tmp, L".srt");
            out_path = tmp;
            make_dirs_w(g_opts.output_dir_w);
        } else {
            out_path = xwcsdup(srt_path);
        }

        // ---- extract audio ----
        info("    extracting audio...");
        fflush(stdout);

        double duration = probe_duration(ffprobe, video);
        float * samples = NULL;
        size_t  n_samples = 0;
        double  extract_secs = 0.0;
        char    err[2048] = { 0 };

        int rc = extract_audio(ffmpeg, video, g_opts.audio_stream, &samples, &n_samples, &extract_secs, err, sizeof(err));
        if (rc != 0) {
            info("\r    audio extraction failed: %s\n", err);
            n_failed++;
            free(video8); free(name8); free(srt_path); free(out_path);
            continue;
        }

        if (!g_opts.no_norm) normalize_rms(samples, n_samples);

        double audio_secs = (double) n_samples / WHISPER_SAMPLE_RATE;
        {
            char durbuf[32] = "";
            if (duration > 0) {
                int dmin = (int) (duration / 60);
                int dsec = (int) duration % 60;
                snprintf(durbuf, sizeof(durbuf), ", stream %.2d:%.2d", dmin, dsec);
            }
            info("\r    audio: %.1f min decoded in %.1fs (%.1f MB)%s\n",
                 audio_secs / 60.0, extract_secs, (double) n_samples * 4.0 / 1048576.0, durbuf);
        }

        if (audio_secs < 1.0) {
            info("    skipping (audio too short)\n");
            n_failed++;
            free(samples); free(video8); free(name8); free(srt_path); free(out_path);
            continue;
        }

        if (n_samples > 2000000000u) {
            info("    audio too long, skipping\n");
            n_failed++;
            free(samples); free(video8); free(name8); free(srt_path); free(out_path);
            continue;
        }

        // ---- transcribe ----
        // Preferred path: one VAD pass over the whole file, then transcribe
        // speech blocks (<= chunk_secs) with fresh decoder context each. This
        // keeps conditioning benefits (punctuation, style) inside a block
        // while making hallucination loops unable to survive past a block.
        struct whisper_full_params wparams = whisper_full_default_params(
            g_opts.beam > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
        wparams.beam_search.beam_size = g_opts.beam;
        wparams.n_threads          = g_opts.threads;
        wparams.translate          = g_opts.translate ? true : false;
        wparams.language           = g_opts.language;
        wparams.print_progress     = false;
        wparams.print_realtime     = false;
        wparams.print_timestamps   = false;
        wparams.print_special      = false;
        wparams.suppress_nst       = true;
        wparams.token_timestamps   = false;
        wparams.initial_prompt     = g_opts.prompt;
        wparams.carry_initial_prompt = g_opts.prompt ? true : false;
        wparams.vad                = false;
        wparams.abort_callback     = abort_cb;
        wparams.new_segment_callback   = new_segment_cb;
        wparams.progress_callback      = progress_cb;

        struct progress_state ps;
        ps.last_pct = -1;
        ps.last_tick = 0;
        ps.base_ms = 0;
        ps.dur_ms = (int64_t) audio_secs * 1000;
        ps.total_ms = ps.dur_ms;
        wparams.progress_callback_user_data = &ps;

        struct cue_list cues;
        cues.items = NULL; cues.count = 0; cues.cap = 0;

        info("    transcribing");
        DWORD t0 = GetTickCount();

        const int64_t total_samples = (int64_t) n_samples;
        const int64_t chunk_samples = (int64_t) g_opts.chunk_secs * WHISPER_SAMPLE_RATE;

        int n_blocks = 0;
        int ret = 0;

        if (vad_ctx) {
            // --- one VAD pass over the whole file ---
            if (!whisper_vad_detect_speech(vad_ctx, samples, (int) n_samples)) {
                info("\r    VAD failed - falling back to single pass\n");
                vad_ctx = NULL; // note: vad_ctx leaked deliberately (init would fail again anyway)
            } else {
                wparams.no_context = true;   // fresh context per block
                struct whisper_vad_segments * segs = whisper_vad_segments_from_probs(vad_ctx, vadp);
                int nseg = segs ? whisper_vad_segments_n_segments(segs) : 0;
                info("\r    VAD: %d speech segment(s)%s\n", nseg, segs ? "" : " (NULL!)");

                int64_t b_start = -1, b_end = -1;
                const int64_t gap_break = 3 * WHISPER_SAMPLE_RATE; // scene/song breaks reset style
                for (int i = 0; i < nseg; i++) {
                    // getters return centiseconds
                    int64_t s = (int64_t) (whisper_vad_segments_get_segment_t0(segs, i) * (WHISPER_SAMPLE_RATE / 100));
                    int64_t e = (int64_t) (whisper_vad_segments_get_segment_t1(segs, i) * (WHISPER_SAMPLE_RATE / 100));
                    if (e > total_samples) e = total_samples;
                    if (e - s < WHISPER_SAMPLE_RATE / 4) continue; // <250 ms, skip
                    if (b_start >= 0 && (e - b_start > chunk_samples || s - b_end > gap_break)) {
                        ret = transcribe_range(ctx, wparams, samples, b_start, b_end, &ps,
                                               b_start * 1000 / WHISPER_SAMPLE_RATE, &cues);
                        n_blocks++;
                        if (ret != 0 || g_abort) break;
                        b_start = -1;
                    }
                    if (b_start < 0) {
                        b_start = s;
                    }
                    b_end = e;
                }
                if (ret == 0 && !g_abort && b_start >= 0) {
                    ret = transcribe_range(ctx, wparams, samples, b_start, b_end, &ps,
                                           b_start * 1000 / WHISPER_SAMPLE_RATE, &cues);
                    n_blocks++;
                }
                if (segs) whisper_vad_free_segments(segs);
            }
        }

        if (!vad_ctx) {
            // --- single pass over the whole file (internal VAD optional) ---
            wparams.vad        = vad8 ? true : false;
            wparams.vad_model_path = vad8;
            wparams.no_context = false;
            ret = transcribe_range(ctx, wparams, samples, 0, total_samples, &ps, 0, &cues);
            n_blocks = 1;
        }

        double proc_secs = (GetTickCount() - t0) / 1000.0;

        // ---- repair decoder loops: re-transcribe the affected span fresh ----
        // Echo-y audio (dream sequences, radio voices) can make the decoder
        // cycle through the same phrase sequence. A fresh mini-pass over just
        // that span decodes it correctly, as the context chain is what sustains
        // the loop.
        wparams.no_context = true;
        for (int attempt = 0; attempt < 8; attempt++) {
            int si, ei;
            if (!find_cycle(&cues, 0, &si, &ei)) break;
            if (ei - si < 4) break; // trivial, leave to postprocess

            int64_t span_t0 = cues.items[si].t0;
            int64_t span_t1 = cues.items[ei - 1].t1;
            if (g_verbose) {
                info("    [cycle %d] cues %d..%d, span %.1fs -> %.1fs\n",
                     attempt, si, ei, span_t0 / 1000.0, span_t1 / 1000.0);
            }
            int64_t ms0 = span_t0 > 2000 ? span_t0 - 2000 : 0;
            int64_t ms1 = span_t1 + 1200;
            int64_t s0 = ms0 * WHISPER_SAMPLE_RATE / 1000;
            int64_t s1 = ms1 * WHISPER_SAMPLE_RATE / 1000;
            if (s1 > total_samples) s1 = total_samples;
            if (s1 - s0 < 2 * WHISPER_SAMPLE_RATE) break;

            struct cue_list retry;
            retry.items = NULL; retry.count = 0; retry.cap = 0;
            if (transcribe_range(ctx, wparams, samples, s0, s1, &ps,
                                 s0 * 1000 / WHISPER_SAMPLE_RATE, &retry) == 0 && retry.count > 0) {
                if (g_verbose) {
                    info("    [retry] %d cues:", retry.count);
                    for (int q = 0; q < retry.count && q < 6; q++) {
                        info(" [%.2f->%.2f %.20s]", retry.items[q].t0 / 1000.0, retry.items[q].t1 / 1000.0, retry.items[q].text);
                    }
                    info("\n");
                }
                int rs, re;
                if (!find_cycle(&retry, 0, &rs, &re)) {
                    cue_splice_replace(&cues, ms0, ms1, &retry);
                    info("\r    repaired decoder loop @%02d:%02d (re-transcribed)\n",
                         (int) (span_t0 / 60000), (int) (span_t0 / 1000 % 60));
                    continue;
                }
            }
            cue_list_free(&retry);
            break; // retry also loops (or failed) - let postprocess collapse it
        }

        cue_postprocess(&cues);

        if (ret != 0 || g_abort) {
            info("\r    transcription failed (code %d)\n", ret);
            n_failed++;
            cue_list_free(&cues);
            free(samples); free(video8); free(name8); free(srt_path); free(out_path);
            if (g_abort) break;
            continue;
        }

        int lang_id = whisper_full_lang_id(ctx);
        const char * lang = lang_id >= 0 ? whisper_lang_str(lang_id) : "?";

        info("\r    transcribed in %.1fs (%.1fx realtime, %d block(s), %d segments)\n",
             proc_secs,
             proc_secs > 0 ? audio_secs / proc_secs : 0.0,
             n_blocks, cues.count);

        // ---- write srt ----
        int written = write_srt(out_path, &cues);
        if (written <= 0) {
            info("    no speech detected - no subtitles written\n");
            n_failed++;
        } else {
            char * out8 = utf16_to_utf8(out_path);
            info("    language: %s%s, %d cues written\n",
                 lang ? lang : "?", g_opts.translate ? " (translated)" : "", written);
            if (g_verbose) info("    -> %s\n", out8);
            free(out8);
            n_done++;
            total_audio_secs += audio_secs;
            total_proc_secs  += proc_secs;
        }

        cue_list_free(&cues);
        free(samples);
        free(video8);
        free(name8);
        free(srt_path);
        free(out_path);
    }

    // ---- summary ----
    double total_secs = (GetTickCount() - t_start) / 1000.0;
    info("\n");
    if (g_abort) info("interrupted.\n");
    info("done: %d processed, %d failed, %d skipped\n", n_done, n_failed, n_skip);
    if (total_proc_secs > 0) {
        info("      %.1f min of audio transcribed in %.1f min (%.1fx realtime)\n",
             total_audio_secs / 60.0, total_proc_secs / 60.0, total_audio_secs / total_proc_secs);
    }
    info("      total wall time: %.1f min\n", total_secs / 60.0);

    whisper_free(ctx);
    if (vad_ctx) whisper_vad_free(vad_ctx);

    return (n_failed == 0 && !g_abort) ? 0 : (g_abort ? 130 : 2);
}
