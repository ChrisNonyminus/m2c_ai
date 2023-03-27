from collections import OrderedDict
import enum
import logging
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
import time
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Union
import os
ASMDIFF_FLAG_PREFIX = "-DIFF"
WINE: str
if "microsoft" in os.uname().release.lower():
    logger.info("WSL detected & nsjail disabled: wine not required.")
    WINE = ""
else:
    WINE = "wine"


@dataclass(frozen=True)
class Checkbox:
    id: str
    flag: str

    def to_json(self) -> Dict[str, str]:
        return {
            "type": "checkbox",
            "id": self.id,
            "flag": self.flag,
        }


@dataclass(frozen=True)
class FlagSet:
    id: str
    flags: List[str]

    def to_json(self) -> Dict[str, Union[str, List[str]]]:
        return {
            "type": "flagset",
            "id": self.id,
            "flags": self.flags,
        }


Flags = List[Union[Checkbox, FlagSet]]


COMMON_GCC_FLAGS: Flags = [
    FlagSet(id="gcc_opt_level", flags=["-O0", "-O1", "-O2", "-O3"]),
    FlagSet(
        id="gcc_debug_level", flags=["-gdwarf-2", "-gdwarf", "-g0", "-g1", "-g2", "-g3"]
    ),
    FlagSet(id="gcc_char_type", flags=["-fsigned-char", "-funsigned-char"]),
    Checkbox("gcc_force_addr", "-fforce-addr"),
]

COMMON_IDO_FLAGS: Flags = [
    FlagSet(id="ido_opt_level", flags=["-O0", "-O1", "-O2", "-O3"]),
    FlagSet(id="ido_debug_level", flags=["-g0", "-g1", "-g2", "-g3"]),
    FlagSet(id="mips_version", flags=["-mips1", "-mips2", "-mips3"]),
    Checkbox("kpic", "-KPIC"),
    Checkbox("pass", "-v"),
]

COMMON_DIFF_FLAGS: Flags = [
    FlagSet(
        id="diff_algorithm",
        flags=[ASMDIFF_FLAG_PREFIX + "levenshtein", ASMDIFF_FLAG_PREFIX + "difflib"],
    ),
]

COMMON_MIPS_DIFF_FLAGS: Flags = [
    Checkbox("mreg_names=32", "-Mreg-names=32"),
    Checkbox("mno_aliases", "-Mno-aliases"),
    Checkbox("no_show_rodata_refs", ASMDIFF_FLAG_PREFIX + "no_show_rodata_refs"),
]

COMMON_GCC_PS1_FLAGS: Flags = [
    FlagSet(id="psyq_opt_level", flags=["-O0", "-O1", "-O2", "-O3", "-Os"]),
    FlagSet(id="gcc_debug_level", flags=["-g0", "-g1", "-g2", "-g3"]),
    FlagSet(id="gcc_char_type", flags=["-fsigned-char", "-funsigned-char"]),
    FlagSet(id="sdata_limit", flags=["-G0", "-G4", "-G8"]),
    FlagSet(id="endianness", flags=["-mel", "-meb"]),
]

@dataclass(frozen=True)
class Platform:
    id: str
    name: str
    description: str
    arch: str
    assemble_cmd: str
    objdump_cmd: str
    nm_cmd: str
    asm_prelude: str
    diff_flags: Flags = field(default_factory=lambda: COMMON_DIFF_FLAGS, hash=False)
    supports_objdump_disassemble: bool = False  # TODO turn into objdump flag


def platform_from_id(platform_id: str) -> Platform:
    if platform_id not in _platforms:
        raise Exception(f"Unknown platform: {platform_id}")
    return _platforms[platform_id]


N64 = Platform(
    id="n64",
    name="Nintendo 64",
    description="MIPS (big-endian)",
    arch="mips",
    assemble_cmd='mips-linux-gnu-as -march=vr4300 -mabi=32 -o "$OUTPUT" "$INPUT"',
    objdump_cmd="mips-linux-gnu-objdump",
    nm_cmd="mips-linux-gnu-nm",
    diff_flags=COMMON_DIFF_FLAGS + COMMON_MIPS_DIFF_FLAGS,
    asm_prelude="""
.macro .late_rodata
    .section .rodata
.endm
.macro .late_rodata_alignment align
.endm
.macro glabel label
    .global \label
    .type \label, @function
    \label:
.endm
.macro dlabel label
    .global \label
    \label:
.endm
.macro jlabel label
    \label:
.endm
.set noat
.set noreorder
.set gp=64
# Float register aliases (o32 ABI)
.set $fv0,          $f0
.set $fv0f,         $f1
.set $fv1,          $f2
.set $fv1f,         $f3
.set $ft0,          $f4
.set $ft0f,         $f5
.set $ft1,          $f6
.set $ft1f,         $f7
.set $ft2,          $f8
.set $ft2f,         $f9
.set $ft3,          $f10
.set $ft3f,         $f11
.set $fa0,          $f12
.set $fa0f,         $f13
.set $fa1,          $f14
.set $fa1f,         $f15
.set $ft4,          $f16
.set $ft4f,         $f17
.set $ft5,          $f18
.set $ft5f,         $f19
.set $fs0,          $f20
.set $fs0f,         $f21
.set $fs1,          $f22
.set $fs1f,         $f23
.set $fs2,          $f24
.set $fs2f,         $f25
.set $fs3,          $f26
.set $fs3f,         $f27
.set $fs4,          $f28
.set $fs4f,         $f29
.set $fs5,          $f30
.set $fs5f,         $f31
""",
)


PS1 = Platform(
    id="ps1",
    name="PlayStation",
    description="MIPS (little-endian)",
    arch="mipsel",
    assemble_cmd='mips-linux-gnu-as -march=r3000 -mabi=32 -o "$OUTPUT" "$INPUT"',
    objdump_cmd="mips-linux-gnu-objdump",
    nm_cmd="mips-linux-gnu-nm",
    diff_flags=COMMON_DIFF_FLAGS + COMMON_MIPS_DIFF_FLAGS,
    asm_prelude="""
.macro .late_rodata
    .section .rodata
.endm
.macro glabel label
    .global \label
    .type \label, @function
    \label:
.endm
.macro jlabel label
    \label:
.endm
.set noat
.set noreorder
""",
)
_platforms: OrderedDict[str, Platform] = OrderedDict(
    {
        "n64": N64,
        "ps1": PS1,
    }
)
import platform as platform_stdlib
COMPILER_BASE_PATH: Path =Path(os.path.dirname(os.path.realpath(__file__)))

class Language(enum.Enum):
    C = "C"
    OLD_CXX = "C++"
    CXX = "C++"
    PASCAL = "Pascal"

    def get_file_extension(self) -> str:
        return {
            Language.C: "c",
            Language.CXX: "cpp",
            Language.OLD_CXX: "c++",
            Language.PASCAL: "p",
        }[self]



@dataclass(frozen=True)
class Compiler:
    id: str
    cc: str
    platform: Platform
    flags: ClassVar[Flags]
    base_id: Optional[str] = None
    is_gcc: ClassVar[bool] = False
    is_ido: ClassVar[bool] = False
    is_mwcc: ClassVar[bool] = False
    needs_wine = False
    language: Language = Language.C

    @property
    def path(self) -> Path:
        return COMPILER_BASE_PATH / (self.base_id or self.id)

    def available(self) -> bool:
        # consider compiler binaries present if the compiler's directory is found
        return self.path.exists()


@dataclass(frozen=True)
class GCCCompiler(Compiler):
    is_gcc: ClassVar[bool] = True
    flags: ClassVar[Flags] = COMMON_GCC_FLAGS


@dataclass(frozen=True)
class GCCPS1Compiler(GCCCompiler):
    flags: ClassVar[Flags] = COMMON_GCC_PS1_FLAGS


@dataclass(frozen=True)
class IDOCompiler(Compiler):
    is_ido: ClassVar[bool] = True
    flags: ClassVar[Flags] = COMMON_IDO_FLAGS



def compiler_from_id(compiler_id: str) -> Compiler:
    if compiler_id not in _compilers:
        raise Exception(f"Unknown compiler: {compiler_id}")
    return _compilers[compiler_id]


# PS1
PSYQ_MSDOS_CC = (
    'cpp -P "$INPUT" | unix2dos > object.oc && cp ${COMPILER_DIR}/* . && '
    + '(HOME="." dosemu -quiet -dumb -f ${COMPILER_DIR}/dosemurc -K . -E "CC1PSX.EXE -quiet ${COMPILER_FLAGS} -o object.os object.oc") &&'
    + '(HOME="." dosemu -quiet -dumb -f ${COMPILER_DIR}/dosemurc -K . -E "ASPSX.EXE -quiet object.os -o object.oo") && '
    + '${COMPILER_DIR}/psyq-obj-parser object.oo -o "$OUTPUT"'
)
PSYQ_CC = 'cat "$INPUT" | unix2dos | ${WINE} ${COMPILER_DIR}/CC1PSX.EXE -quiet ${COMPILER_FLAGS} -o "$OUTPUT".s && ${WINE} ${COMPILER_DIR}/ASPSX.EXE -quiet "$OUTPUT".s -o "$OUTPUT".obj && ${COMPILER_DIR}/psyq-obj-parser "$OUTPUT".obj -o "$OUTPUT"'

PSYQ35 = GCCPS1Compiler(
    id="psyq3.5",
    platform=PS1,
    cc=PSYQ_MSDOS_CC,
)

PSYQ36 = GCCPS1Compiler(
    id="psyq3.6",
    platform=PS1,
    cc=PSYQ_MSDOS_CC,
)

PSYQ40 = GCCPS1Compiler(
    id="psyq4.0",
    platform=PS1,
    cc=PSYQ_CC,
)

PSYQ41 = GCCPS1Compiler(
    id="psyq4.1",
    platform=PS1,
    cc=PSYQ_CC,
)

PSYQ43 = GCCPS1Compiler(
    id="psyq4.3",
    platform=PS1,
    cc=PSYQ_CC,
)

PSYQ45 = GCCPS1Compiler(
    id="psyq4.5",
    platform=PS1,
    cc=PSYQ_CC,
)

PSYQ46 = GCCPS1Compiler(
    id="psyq4.6",
    platform=PS1,
    cc=PSYQ_CC,
)


# N64
IDO53 = IDOCompiler(
    id="ido5.3",
    platform=N64,
    cc='USR_LIB="${COMPILER_DIR}" "${COMPILER_DIR}/cc" -c -Xcpluscomm -G0 -non_shared -Wab,-r4300_mul -woff 649,838,712 -32 ${COMPILER_FLAGS} -o "${OUTPUT}" "${INPUT}"',
)

IDO53_CXX = IDOCompiler(
    id="ido5.3_c++",
    platform=N64,
    cc='"${COMPILER_DIR}"/usr/bin/qemu-irix -L "${COMPILER_DIR}" "${COMPILER_DIR}/usr/lib/CC" -I "{COMPILER_DIR}"/usr/include -c -Xcpluscomm -G0 -non_shared -woff 649,838,712 -32 ${COMPILER_FLAGS} -o "${OUTPUT}" "${INPUT}"',
    base_id="ido5.3_c++",
    language=Language.OLD_CXX,
)

IDO71 = IDOCompiler(
    id="ido7.1",
    platform=N64,
    cc='USR_LIB="${COMPILER_DIR}" "${COMPILER_DIR}/cc" -c -Xcpluscomm -G0 -non_shared -Wab,-r4300_mul -woff 649,838,712 -32 ${COMPILER_FLAGS} -o "${OUTPUT}" "${INPUT}"',
)

IDO60 = IDOCompiler(
    id="ido6.0",
    platform=N64,
    cc='"${COMPILER_DIR}"/usr/bin/qemu-irix -L "${COMPILER_DIR}" "${COMPILER_DIR}/usr/bin/cc" -c -Xcpluscomm -G0 -non_shared -woff 649,838,712 -32 ${COMPILER_FLAGS} -o "${OUTPUT}" "${INPUT}"',
    base_id="ido6.0",
)

GCC272KMC = GCCCompiler(
    id="gcc2.7.2kmc",
    platform=N64,
    cc='COMPILER_PATH="${COMPILER_DIR}" "${COMPILER_DIR}"/gcc -c -G0 -mgp32 -mfp32 ${COMPILER_FLAGS} "${INPUT}" -o "${OUTPUT}"',
)

GCC281 = GCCCompiler(
    id="gcc2.8.1",
    platform=N64,
    cc='"${COMPILER_DIR}"/gcc -G0 -c -B "${COMPILER_DIR}"/ $COMPILER_FLAGS "$INPUT" -o "$OUTPUT"',
)

GCC272SN = GCCCompiler(
    id="gcc2.7.2sn",
    platform=N64,
    cc='cpp -P "$INPUT" | ${WINE} "${COMPILER_DIR}"/cc1n64.exe -quiet -G0 -mcpu=vr4300 -mips3 -mhard-float -meb ${COMPILER_FLAGS} -o "$OUTPUT".s && ${WINE} "${COMPILER_DIR}"/asn64.exe -q -G0 "$OUTPUT".s -o "$OUTPUT".obj && "${COMPILER_DIR}"/psyq-obj-parser "$OUTPUT".obj -o "$OUTPUT" -b -n',
)

GCC272SNEW = GCCCompiler(
    id="gcc2.7.2snew",
    platform=N64,
    cc='"${COMPILER_DIR}"/cpp -lang-c -undef "$INPUT" | "${COMPILER_DIR}"/cc1 -mfp32 -mgp32 -G0 -quiet -mcpu=vr4300 -fno-exceptions ${COMPILER_FLAGS} -o "$OUTPUT".s && python3 "${COMPILER_DIR}"/modern-asn64.py mips-linux-gnu-as "$OUTPUT".s -G0 -EB -mips3 -O1 -mabi=32 -mgp32 -march=vr4300 -mfp32 -mno-shared -o "$OUTPUT"',
)

GCC281SNCXX = GCCCompiler(
    id="gcc2.8.1sn-cxx",
    base_id="gcc2.8.1sn",
    platform=N64,
    cc='cpp -E -lang-c++ -undef -D__GNUC__=2 -D__cplusplus -Dmips -D__mips__ -D__mips -Dn64 -D__n64__ -D__n64 -D_PSYQ -D__EXTENSIONS__ -D_MIPSEB -D__CHAR_UNSIGNED__ -D_LANGUAGE_C_PLUS_PLUS "$INPUT" '
    '| ${WINE} "${COMPILER_DIR}"/cc1pln64.exe ${COMPILER_FLAGS} -o "$OUTPUT".s '
    '&& ${WINE} "${COMPILER_DIR}"/asn64.exe -q -G0 "$OUTPUT".s -o "$OUTPUT".obj '
    '&& "${COMPILER_DIR}"/psyq-obj-parser "$OUTPUT".obj -o "$OUTPUT" -b -n',
)

EGCS1124 = GCCCompiler(
    id="egcs_1.1.2-4",
    platform=N64,
    cc='COMPILER_PATH="${COMPILER_DIR}" "${COMPILER_DIR}"/mips-linux-gcc -c -G 0 -fno-PIC -mgp32 -mfp32 -mcpu=4300 -nostdinc ${COMPILER_FLAGS} "${INPUT}" -o "${OUTPUT}"',
)



_all_compilers: List[Compiler] = [
    # PS1
    PSYQ35,
    PSYQ36,
    PSYQ40,
    PSYQ41,
    PSYQ43,
    PSYQ45,
    PSYQ46,
    # N64
    IDO53,
    IDO53_CXX,
    IDO60,
    IDO71,
    GCC272KMC,
    GCC272SN,
    GCC272SNEW,
    GCC281,
    GCC281SNCXX,
    EGCS1124,
]
_compilers = OrderedDict({c.id: c for c in _all_compilers if c.available()})



@dataclass
class CompilationResult:
    elf_object: bytes
    errors: str

from subprocess import CalledProcessError
class SubprocessError(Exception):
    SUBPROCESS_NAME: ClassVar[str] = "Subprocess"
    msg: str
    stdout: str
    stderr: str

    def __init__(self, message: str):
        self.msg = f"{self.SUBPROCESS_NAME} error: {message}"

        super().__init__(self.msg)
        self.stdout = ""
        self.stderr = ""

    @staticmethod
    def from_process_error(ex: CalledProcessError) -> "SubprocessError":
        error = SubprocessError(f"{ex.cmd[0]} returned {ex.returncode}")
        error.stdout = ex.stdout
        error.stderr = ex.stderr
        error.msg = ex.stdout
        return error


class DiffError(SubprocessError):
    SUBPROCESS_NAME: ClassVar[str] = "Diff"


class ObjdumpError(SubprocessError):
    SUBPROCESS_NAME: ClassVar[str] = "objdump"


class NmError(SubprocessError):
    SUBPROCESS_NAME: ClassVar[str] = "nm"


class CompilationError(SubprocessError):
    SUBPROCESS_NAME: ClassVar[str] = "Compiler"


class SandboxError(SubprocessError):
    SUBPROCESS_NAME: ClassVar[str] = "Sandbox"
class Asm():
    hash = ""
    data = ""

    def __str__(self) -> str:
        return self.data if len(self.data) < 20 else self.data[:17] + "..."


class Assembly():
    hash = ""
    time = time.time()
    arch = ""
    source_asm = ""
    elf_object = bytearray()

DiffResult = Dict[str, Any]
class AssemblyError(SubprocessError):
    SUBPROCESS_NAME: ClassVar[str] = "Compiler"

    @staticmethod
    def from_process_error(ex: CalledProcessError) -> "SubprocessError":
        error = super(AssemblyError, AssemblyError).from_process_error(ex)

        error_lines = []
        for line in ex.stdout.splitlines():
            if "asm.s:" in line:
                error_lines.append(line[line.find("asm.s:") + len("asm.s:") :].strip())
            else:
                error_lines.append(line)
        error.msg = "\n".join(error_lines)

        return error

class CompilerWrapper:
    @staticmethod
    def filter_compiler_flags(compiler_flags: str) -> str:
        # Remove irrelevant flags that are part of the base compiler configs or
        # don't affect matching, but clutter the compiler settings field.
        # TODO: use cfg for this?
        skip_flags_with_args = {
            "-B",
            "-I",
            "-D",
            "-U",
        }
        skip_flags = {
            "-ffreestanding",
            "-non_shared",
            "-Xcpluscomm",
            "-Wab,-r4300_mul",
            "-c",
        }

        skip_next = False
        flags = []
        for flag in compiler_flags.split():
            if skip_next:
                skip_next = False
                continue
            if flag in skip_flags:
                continue
            if flag in skip_flags_with_args:
                skip_next = True
                continue
            if any(flag.startswith(f) for f in skip_flags_with_args):
                continue
            flags.append(flag)
        return " ".join(flags)

    @staticmethod
    def filter_compile_errors(input: str) -> str:
        if input is None:
            return ""
        filter_strings = [
            r"wine: could not load .*\.dll.*\n?",
            r"wineserver: could not save registry .*\n?",
            r"### .*\.exe Driver Error:.*\n?",
            r"#   Cannot find my executable .*\n?",
            r"### MWCPPC\.exe Driver Error:.*\n?",
        ]

        for str in filter_strings:
            input = re.sub(str, "", input)

        return input.strip()

    @staticmethod
    def compile_code(
        compiler: Compiler,
        compiler_flags: str,
        code: str,
        context: str,
        function: str = "",
    ) -> CompilationResult:

        code = code.replace("\r\n", "\n")
        context = context.replace("\r\n", "\n")

        sandbox_temp_dir = "work/"
        sandbox_path = Path(sandbox_temp_dir)
        if True:
            ext = compiler.language.get_file_extension()
            code_file = f"code.{ext}"
            ctx_file = f"ctx.{ext}"

            code_path = sandbox_path / code_file
            object_path = sandbox_path / "object.o"
            with code_path.open("w") as f:
                f.write(f'#line 1 "{ctx_file}"\n')
                f.write(context)
                f.write("\n")

                f.write(f'#line 1 "{code_file}"\n')
                f.write(code)
                f.write("\n")

            cc_cmd = compiler.cc

            # Fix for MWCC line numbers in GC 3.0+
            if compiler.is_mwcc:
                ctx_path = sandbox_path / ctx_file
                ctx_path.touch()

            # IDO hack to support -KPIC
            if compiler.is_ido and "-KPIC" in compiler_flags:
                cc_cmd = cc_cmd.replace("-non_shared", "")

            # Run compiler
            try:
                st = round(time.time() * 1000)
                print (cc_cmd)
                compile_proc = subprocess.run(
                    [cc_cmd],
                    shell=True,
                    env={
                        "WINE": WINE,
                        "INPUT": (code_path),
                        "OUTPUT": (object_path),
                        "COMPILER_DIR": (compiler.path),
                        "COMPILER_FLAGS": (compiler_flags),
                        "FUNCTION": function,
                        "MWCIncludes": "/tmp",
                        "TMPDIR": "/tmp",
                    }, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
                et = round(time.time() * 1000)
                logging.debug(f"Compilation finished in: {et - st} ms")
            except subprocess.CalledProcessError as e:
                # Compilation failed
                msg = e.stdout

                logging.debug("Compilation failed: %s", msg)
                raise CompilationError(CompilerWrapper.filter_compile_errors(msg))
            except ValueError as e:
                # Shlex issue?
                logging.debug("Compilation failed: %s", e)
                raise CompilationError(str(e))
            except subprocess.TimeoutExpired as e:
                raise CompilationError("Compilation failed: timeout expired")

            if not object_path.exists():
                error_msg = (
                    "Compiler did not create an object file: %s" % compile_proc.stdout
                )
                logging.debug(error_msg)
                raise CompilationError(error_msg)

            object_bytes = object_path.read_bytes()

            if not object_bytes:
                raise CompilationError("Compiler created an empty object file")

            compile_errors = CompilerWrapper.filter_compile_errors(compile_proc.stdout.decode("utf-8"))

            return CompilationResult(object_path.read_bytes(), compile_errors)
