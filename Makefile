# imgedit - Linux / macOS build. Windows: use build.ps1.
#   make            command-line tool -> build/imgedit
#   make wasm       web/imgedit.wasm (needs zig, or clang with a wasi sysroot: WASM_CC=...)
#   make test       unit tests (and the wasm tests when node is available)
#   make runtime    download ONNX Runtime next to build/imgedit

CC      ?= cc
CFLAGS  ?= -O3
WARN     = -Wall -Wextra -Wno-unused-function
WASM_CC ?= zig cc -target wasm32-wasi
ORT_VERSION ?= 1.31.0

LIB = src/parallel.c src/image.c src/png.c src/color.c src/geometry.c src/filter.c src/matte.c
CLI = src/nn.c src/platform.c src/cli.c
UNAME := $(shell uname -s)
LIBS = -lm -lpthread
ifeq ($(UNAME),Linux)
LIBS += -ldl
endif

all: build/imgedit

build/imgedit: $(LIB) $(CLI) src/*.h
	@mkdir -p build
	$(CC) -std=c11 $(CFLAGS) $(WARN) $(LIB) $(CLI) $(LIBS) -o $@

wasm: web/imgedit.wasm

web/imgedit.wasm: $(LIB) src/wasm.c src/*.h
	$(WASM_CC) -std=c11 -O3 -msimd128 -mbulk-memory -mexec-model=reactor -DIE_NO_FILES -s \
		-Wl,--max-memory=4294901760 -Wl,-z,stack-size=1048576 $(WARN) $(LIB) src/wasm.c -o $@

build/test_core: $(LIB) tests/test_core.c src/*.h
	@mkdir -p build
	$(CC) -std=c11 -O2 $(WARN) $(LIB) tests/test_core.c $(LIBS) -o $@

test: build/test_core
	./build/test_core
	@if [ -f web/imgedit.wasm ] && command -v node >/dev/null; then node tests/wasm.test.mjs; fi

runtime:
	@mkdir -p build
ifeq ($(UNAME),Darwin)
	curl -fL https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)/onnxruntime-osx-arm64-$(ORT_VERSION).tgz | tar xz -C build
	cp build/onnxruntime-osx-arm64-$(ORT_VERSION)/lib/libonnxruntime*.dylib build/
	cd build && ln -sf libonnxruntime.$(ORT_VERSION).dylib libonnxruntime.dylib
else
	curl -fL https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)/onnxruntime-linux-x64-$(ORT_VERSION).tgz | tar xz -C build
	cp -P build/onnxruntime-linux-x64-$(ORT_VERSION)/lib/libonnxruntime.so* build/
endif

clean:
	rm -rf build web/imgedit.wasm

.PHONY: all wasm test runtime clean
