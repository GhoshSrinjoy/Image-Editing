/* POSIX and Darwin declarations (clock_gettime, readlink, realpath) under -std=c11. */
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE 1
#define _DARWIN_C_SOURCE 1
#endif

#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static int push_name(char ***names, int *count, int *cap, const char *s)
{
    if (*count == *cap) {
        int nc = *cap ? *cap * 2 : 64;
        char **t = (char **)realloc(*names, sizeof(char *) * (size_t)nc);
        if (!t) return -1;
        *names = t;
        *cap = nc;
    }
    size_t n = strlen(s) + 1;
    char *d = (char *)malloc(n);
    if (!d) return -1;
    memcpy(d, s, n);
    (*names)[(*count)++] = d;
    return 0;
}

void pf_free_list(char **names, int count)
{
    for (int i = 0; i < count; i++) free(names[i]);
    free(names);
}

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>

#define WPATH 4096

static int to_w(const char *s, wchar_t *w, int n) { return MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n) > 0; }
static int to_u8(const wchar_t *w, char *s, int n) { return WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL) > 0; }

double pf_now(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

int pf_exe_dir(char *buf, size_t n)
{
    wchar_t w[WPATH];
    DWORD len = GetModuleFileNameW(NULL, w, WPATH);
    if (!len || len >= WPATH) return -1;
    while (len > 0 && w[len - 1] != L'\\' && w[len - 1] != L'/') len--;
    if (len > 0) len--;
    w[len] = 0;
    return to_u8(w, buf, (int)n) ? 0 : -1;
}

const char *pf_getenv(const char *name)
{
    static char out[8][WPATH];
    static int slot;
    wchar_t wn[256], *v;
    if (!to_w(name, wn, 256) || !(v = _wgetenv(wn)) || !*v) return NULL;
    char *o = out[slot++ & 7];
    return to_u8(v, o, WPATH) ? o : NULL;
}

int pf_home_dir(char *buf, size_t n)
{
    const char *h = pf_getenv("USERPROFILE");
    if (!h || strlen(h) + 1 > n) return -1;
    memcpy(buf, h, strlen(h) + 1);
    return 0;
}

static DWORD attrs(const char *path)
{
    wchar_t w[WPATH];
    return to_w(path, w, WPATH) ? GetFileAttributesW(w) : INVALID_FILE_ATTRIBUTES;
}

int pf_exists(const char *path) { return attrs(path) != INVALID_FILE_ATTRIBUTES; }

int pf_is_dir(const char *path)
{
    DWORD a = attrs(path);
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

int pf_mkdirs(const char *path)
{
    char tmp[WPATH];
    size_t n = strlen(path);
    if (n >= sizeof tmp) return -1;
    memcpy(tmp, path, n + 1);
    for (size_t i = 1; i <= n; i++) {
        if (tmp[i] != '\\' && tmp[i] != '/' && tmp[i] != 0) continue;
        if (i == 2 && tmp[1] == ':') continue; /* drive root */
        char c = tmp[i];
        tmp[i] = 0;
        wchar_t w[WPATH];
        if (to_w(tmp, w, WPATH) && !CreateDirectoryW(w, NULL) && GetLastError() != ERROR_ALREADY_EXISTS) {
            tmp[i] = c;
            return -1;
        }
        tmp[i] = c;
    }
    return 0;
}

int pf_rename(const char *from, const char *to)
{
    wchar_t a[WPATH], b[WPATH];
    if (!to_w(from, a, WPATH) || !to_w(to, b, WPATH)) return -1;
    return MoveFileExW(a, b, MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED) ? 0 : -1;
}

int pf_remove(const char *path)
{
    wchar_t w[WPATH];
    return to_w(path, w, WPATH) && DeleteFileW(w) ? 0 : -1;
}

int pf_list_dir(const char *dir, char ***names, int *count)
{
    *names = NULL;
    *count = 0;
    int cap = 0;
    char pattern[WPATH];
    snprintf(pattern, sizeof pattern, "%s\\*", dir);
    wchar_t w[WPATH];
    if (!to_w(pattern, w, WPATH)) return -1;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(w, &fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        char name[WPATH];
        if (to_u8(fd.cFileName, name, WPATH) && push_name(names, count, &cap, name)) break;
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    if (*count) qsort(*names, (size_t)*count, sizeof(char *), cmp_str);
    return 0;
}

/* Quote one argument following the rules CommandLineToArgvW uses. */
static void quote_arg(const wchar_t *a, wchar_t *out, size_t *len, size_t cap)
{
#define PUT(c) \
    do { \
        if (*len + 1 < cap) out[(*len)++] = (c); \
    } while (0)
    int need = !*a || wcspbrk(a, L" \t\"") != NULL;
    if (*len) PUT(L' ');
    if (!need) {
        for (; *a; a++) PUT(*a);
        return;
    }
    PUT(L'"');
    for (;; a++) {
        unsigned bs = 0;
        while (*a == L'\\') a++, bs++;
        if (!*a) {
            for (unsigned i = 0; i < bs * 2; i++) PUT(L'\\');
            break;
        }
        if (*a == L'"') {
            for (unsigned i = 0; i < bs * 2 + 1; i++) PUT(L'\\');
        } else {
            for (unsigned i = 0; i < bs; i++) PUT(L'\\');
        }
        PUT(*a);
    }
    PUT(L'"');
#undef PUT
}

int pf_run(const char *const *argv)
{
    static wchar_t cmd[32768];
    size_t len = 0;
    for (int i = 0; argv[i]; i++) {
        wchar_t w[WPATH];
        if (!to_w(argv[i], w, WPATH)) return -1;
        quote_arg(w, cmd, &len, sizeof cmd / sizeof cmd[0]);
    }
    cmd[len] = 0;
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    if (!CreateProcessW(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) return -1;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)code;
}

void pf_init_args(int *argc, char ***argv)
{
    SetConsoleOutputCP(CP_UTF8);
    int n = 0;
    wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), &n);
    if (!wargv) return;
    char **out = (char **)calloc((size_t)n + 1, sizeof(char *));
    if (!out) {
        LocalFree(wargv);
        return;
    }
    for (int i = 0; i < n; i++) {
        int need = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
        out[i] = (char *)malloc((size_t)(need > 0 ? need : 1));
        if (!out[i] || need <= 0 || !WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, out[i], need, NULL, NULL)) {
            LocalFree(wargv);
            return; /* keep the original argv */
        }
    }
    LocalFree(wargv);
    *argc = n;
    *argv = out; /* lives for the whole run */
}

#else /* POSIX */

#include <dirent.h>
#include <errno.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

extern char **environ;

double pf_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int pf_exe_dir(char *buf, size_t n)
{
    char path[4096];
#ifdef __APPLE__
    uint32_t size = sizeof path;
    if (_NSGetExecutablePath(path, &size) != 0) return -1;
    char real[4096];
    if (realpath(path, real)) memcpy(path, real, strlen(real) + 1);
#else
    ssize_t len = readlink("/proc/self/exe", path, sizeof path - 1);
    if (len <= 0) return -1;
    path[len] = 0;
#endif
    char *slash = strrchr(path, '/');
    if (!slash) return -1;
    *slash = 0;
    if (strlen(path) + 1 > n) return -1;
    memcpy(buf, path, strlen(path) + 1);
    return 0;
}

const char *pf_getenv(const char *name)
{
    const char *v = getenv(name);
    return v && *v ? v : NULL;
}

int pf_home_dir(char *buf, size_t n)
{
    const char *h = pf_getenv("HOME");
    if (!h || strlen(h) + 1 > n) return -1;
    memcpy(buf, h, strlen(h) + 1);
    return 0;
}

int pf_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

int pf_is_dir(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

int pf_mkdirs(const char *path)
{
    char tmp[4096];
    size_t n = strlen(path);
    if (n >= sizeof tmp) return -1;
    memcpy(tmp, path, n + 1);
    for (size_t i = 1; i <= n; i++) {
        if (tmp[i] != '/' && tmp[i] != 0) continue;
        char c = tmp[i];
        tmp[i] = 0;
        if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
        tmp[i] = c;
    }
    return 0;
}

int pf_rename(const char *from, const char *to) { return rename(from, to); }
int pf_remove(const char *path) { return remove(path); }

int pf_list_dir(const char *dir, char ***names, int *count)
{
    *names = NULL;
    *count = 0;
    int cap = 0;
    DIR *d = opendir(dir);
    if (!d) return -1;
    struct dirent *e;
    while ((e = readdir(d))) {
        char full[4096];
        struct stat st;
        snprintf(full, sizeof full, "%s/%s", dir, e->d_name);
        if (stat(full, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        if (push_name(names, count, &cap, e->d_name)) break;
    }
    closedir(d);
    if (*count) qsort(*names, (size_t)*count, sizeof(char *), cmp_str);
    return 0;
}

int pf_run(const char *const *argv)
{
    pid_t pid;
    if (posix_spawnp(&pid, argv[0], NULL, NULL, (char *const *)argv, environ) != 0) return -1;
    int status;
    while (waitpid(pid, &status, 0) < 0)
        if (errno != EINTR) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

void pf_init_args(int *argc, char ***argv)
{
    (void)argc;
    (void)argv;
}

#endif
