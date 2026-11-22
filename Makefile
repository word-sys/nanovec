ifeq ($(origin CC),default)
    CC := gcc
endif
AR        ?= ar
PREFIX    ?= /usr/local
BUILD_DIR ?= build

ARCH := $(shell uname -m)

VERSION_MAJOR := $(shell awk '$$2 == "NANOVEC_VERSION_MAJOR" {print $$3}' include/nanovec_version.h)
VERSION_MINOR := $(shell awk '$$2 == "NANOVEC_VERSION_MINOR" {print $$3}' include/nanovec_version.h)
VERSION_PATCH := $(shell awk '$$2 == "NANOVEC_VERSION_PATCH" {print $$3}' include/nanovec_version.h)
VERSION       := $(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_PATCH)

ENABLE_AVX2   ?= 1
ENABLE_AVX512 ?= 1

BASE_CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -fPIC -fvisibility=hidden -D_GNU_SOURCE
ifeq ($(DEBUG),1)
    OPT_CFLAGS := -O0 -g3 -fno-omit-frame-pointer
else
    OPT_CFLAGS := -O3
endif

ifeq ($(ASAN),1)
    SAN_CFLAGS  := -fsanitize=address,undefined -fno-omit-frame-pointer -g
    SAN_LDFLAGS := -fsanitize=address,undefined
else
    SAN_CFLAGS  :=
    SAN_LDFLAGS :=
endif

CPPFLAGS := -Iinclude -Isrc $(EXTRA_CPPFLAGS)
CFLAGS   := $(BASE_CFLAGS) $(OPT_CFLAGS) $(SAN_CFLAGS) $(EXTRA_CFLAGS)
LDFLAGS  := $(SAN_LDFLAGS) $(EXTRA_LDFLAGS)
LDLIBS   := -lm -lpthread

SRC_DIRS := src/core src/math src/quant src/storage src/index src/inference src/platform
ALL_SRCS := $(foreach d,$(SRC_DIRS),$(wildcard $(d)/*.c))

ARCH_SIMD_SRCS := src/math/simd_avx2.c src/math/simd_avx512.c src/math/simd_neon.c
LIB_SRCS := $(filter-out $(ARCH_SIMD_SRCS),$(ALL_SRCS))

ifneq ($(filter x86_64 amd64,$(ARCH)),)
    ifeq ($(ENABLE_AVX2),1)
        LIB_SRCS += $(wildcard src/math/simd_avx2.c)
    endif
    ifeq ($(ENABLE_AVX512),1)
        LIB_SRCS += $(wildcard src/math/simd_avx512.c)
    endif
else ifneq ($(filter aarch64 arm64 arm%,$(ARCH)),)
    LIB_SRCS += $(wildcard src/math/simd_neon.c)
endif

LIB_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(LIB_SRCS))
DEPS     := $(LIB_OBJS:.o=.d)

LIB_DIR     := $(BUILD_DIR)/lib
STATIC_LIB  := $(LIB_DIR)/libnanovec.a
SHARED_REAL := $(LIB_DIR)/libnanovec.so.$(VERSION)
SHARED_SO   := $(LIB_DIR)/libnanovec.so.$(VERSION_MAJOR)
SHARED_LINK := $(LIB_DIR)/libnanovec.so

TEST_SRCS := $(wildcard tests/test_*.c)
TEST_BINS := $(patsubst tests/%.c,$(BUILD_DIR)/tests/%,$(TEST_SRCS))
DEPS      += $(patsubst tests/%.c,$(BUILD_DIR)/tests/%.d,$(TEST_SRCS))

TOOL_SRCS := $(wildcard tools/*.c)
TOOL_BINS := $(patsubst tools/%.c,$(BUILD_DIR)/bin/%,$(TOOL_SRCS))
DEPS      += $(patsubst tools/%.c,$(BUILD_DIR)/bin/%.d,$(TOOL_SRCS))

HEADERS := $(wildcard include/*.h)

.PHONY: all libs tools tests test check-headers install uninstall clean info

all: libs tools

ifneq ($(strip $(LIB_OBJS)),)
libs: $(STATIC_LIB) $(SHARED_LINK)
else
libs:
	@echo "[nanovec] No library sources yet; skipping library build."
endif

$(BUILD_DIR)/src/math/simd_avx2.o: SIMD_FLAGS := -mavx2 -mfma
$(BUILD_DIR)/src/math/simd_avx512.o: SIMD_FLAGS := -mavx512f -mavx512dq -mavx512bw
ifneq ($(filter arm%,$(ARCH)),)
$(BUILD_DIR)/src/math/simd_neon.o: SIMD_FLAGS := -mfpu=neon
endif

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SIMD_FLAGS) -MMD -MP -c $< -o $@

$(STATIC_LIB): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(SHARED_REAL): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) -shared -Wl,-soname,libnanovec.so.$(VERSION_MAJOR) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(SHARED_SO): $(SHARED_REAL)
	ln -sf $(notdir $<) $@

$(SHARED_LINK): $(SHARED_SO)
	ln -sf $(notdir $<) $@

ifneq ($(strip $(LIB_OBJS)),)
TEST_DEPS := $(STATIC_LIB)
TEST_LIBS := $(STATIC_LIB)
else
TEST_DEPS :=
TEST_LIBS :=
endif

$(BUILD_DIR)/tests/%: tests/%.c $(TEST_DEPS)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -MMD -MP $< $(TEST_LIBS) $(LDLIBS) -o $@

$(BUILD_DIR)/bin/%: tools/%.c $(STATIC_LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -MMD -MP $< $(STATIC_LIB) $(LDLIBS) -o $@

ifneq ($(strip $(TOOL_BINS)),)
tools: libs $(TOOL_BINS)
else
tools:
	@echo "[nanovec] No tool sources yet; skipping tools build."
endif

ifneq ($(strip $(TEST_BINS)),)
tests: libs $(TEST_BINS)
else
tests:
	@echo "[nanovec] No test sources yet; skipping tests build."
endif

test: check-headers tests
	@for t in $(TEST_BINS); do \
		echo "[TEST] $$t"; \
		"$$t" || exit 1; \
	done
	@if [ -f tests/test_integration.py ] && [ -f "$(SHARED_LINK)" ]; then \
		echo "[TEST] tests/test_integration.py"; \
		NANOVEC_LIB="$(CURDIR)/$(SHARED_LINK)" python3 tests/test_integration.py || exit 1; \
	fi

check-headers:
	@for h in $(HEADERS); do \
		echo "[HEADER] $$h"; \
		printf '#include "%s"\n#include "%s"\nint nanovec_hdr_chk_(void);\n' "$$h" "$$h" \
			| $(CC) $(CPPFLAGS) $(CFLAGS) -Werror -fsyntax-only -x c - || exit 1; \
	done

install: libs
	@mkdir -p "$(DESTDIR)$(PREFIX)/include" "$(DESTDIR)$(PREFIX)/lib"
	install -m 644 $(HEADERS) "$(DESTDIR)$(PREFIX)/include/"
	@if [ -f "$(STATIC_LIB)" ]; then \
		install -m 644 "$(STATIC_LIB)" "$(DESTDIR)$(PREFIX)/lib/"; \
	fi
	@if [ -f "$(SHARED_REAL)" ]; then \
		install -m 755 "$(SHARED_REAL)" "$(DESTDIR)$(PREFIX)/lib/"; \
		ln -sf "libnanovec.so.$(VERSION)" "$(DESTDIR)$(PREFIX)/lib/libnanovec.so.$(VERSION_MAJOR)"; \
		ln -sf "libnanovec.so.$(VERSION_MAJOR)" "$(DESTDIR)$(PREFIX)/lib/libnanovec.so"; \
	fi

uninstall:
	rm -f "$(DESTDIR)$(PREFIX)/include/nanovec.h" \
	      "$(DESTDIR)$(PREFIX)/include/nanovec_version.h" \
	      "$(DESTDIR)$(PREFIX)/lib/libnanovec.a" \
	      "$(DESTDIR)$(PREFIX)/lib/libnanovec.so" \
	      "$(DESTDIR)$(PREFIX)/lib/libnanovec.so.$(VERSION_MAJOR)" \
	      "$(DESTDIR)$(PREFIX)/lib/libnanovec.so.$(VERSION)"

clean:
	rm -rf $(BUILD_DIR)

info:
	@echo "nanovec v$(VERSION) ($(ARCH))"
	@echo "CC:       $(CC)"
	@echo "CFLAGS:   $(CFLAGS)"
	@echo "LIB_SRCS: $(LIB_SRCS)"
	@echo "TESTS:    $(TEST_SRCS)"
	@echo "TOOLS:    $(TOOL_SRCS)"

-include $(DEPS)
