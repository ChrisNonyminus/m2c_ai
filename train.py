import json
import random
import gym
import clang
import pycparser
import requests
import jsonpickle
import zipfile
import pickle
from compilers.download import download_n64, download_ps1
from compilers.compiler_defs import *
import asmdiffer.diff as asmdiffer


def scrape_scratches(url = "https://decomp.me/api/scratch", scratches_list : list[tuple] = [], recursion_depth = 768):
    scratches = jsonpickle.decode(requests.get(url).text)
    next_url = scratches['next']
    for result in scratches["results"]:
        if result['platform'] in ['n64', 'psx'] and (result['score'] >= 300 and result['score'] < 1000):
            compiler = result['compiler']
            compiler_flags = jsonpickle.decode(requests.get("https://decomp.me/api/scratch" + "/" + result['slug']).text)['compiler_flags']
            with open("zips/" + result['slug'] + ".zip", 'wb') as f:
                f.write(requests.get("https://decomp.me/api/scratch" + "/" + result['slug'] + "/export").content)
            with zipfile.ZipFile("zips/" + result['slug'] + ".zip", mode="r") as archive:
                ctx_c = archive.read("ctx.c").decode("utf-8") if "ctx.c" in archive.namelist() else ""
                code_c = archive.read('code.c').decode("utf-8")
                target_s = archive.read('target.s').decode('utf-8')
                target_o = archive.read('target.o')
                diff_label = result['name']
                platform = result['platform']
                scratches_list.append((diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o))
    if (recursion_depth > 0):
        scratches_list = scrape_scratches(next_url, scratches_list, recursion_depth - 1)
    return scratches_list

class DiffWrapper:
    @staticmethod
    def filter_objdump_flags(compiler_flags: str) -> str:
        # Remove irrelevant flags that are part of the base objdump configs, but clutter the compiler settings field.
        # TODO: use cfg for this?
        skip_flags_with_args: set[str] = set()
        skip_flags = {
            "--disassemble",
            "--disassemble-zeroes",
            "--line-numbers",
            "--reloc",
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
    def create_config(
        arch: asmdiffer.ArchSettings, diff_flags: List[str]
    ) -> asmdiffer.Config:
        show_rodata_refs = "-DIFFno_show_rodata_refs" not in diff_flags
        algorithm = "difflib" if "-DIFFdifflib" in diff_flags else "levenshtein"

        return asmdiffer.Config(
            arch=arch,
            # Build/objdump options
            diff_obj=True,
            objfile="",
            make=False,
            source_old_binutils=True,
            diff_section=".text",
            inlines=False,
            max_function_size_lines=25000,
            max_function_size_bytes=25000 * 4,
            # Display options
            formatter=asmdiffer.JsonFormatter(arch_str=arch.name),
            diff_mode=asmdiffer.DiffMode.NORMAL,
            base_shift=0,
            skip_lines=0,
            compress=None,
            show_branches=True,
            show_line_numbers=False,
            show_source=False,
            stop_at_ret=False,
            ignore_large_imms=False,
            ignore_addr_diffs=True,
            algorithm=algorithm,
            reg_categories={},
            show_rodata_refs=show_rodata_refs,
        )

    @staticmethod
    def get_objdump_target_function_flags(
        sandbox_path : Path, target_path: Path, platform: Platform, label: str
    ) -> List[str]:
        if not label:
            return ["--start-address=0"]

        if platform.supports_objdump_disassemble:
            return [f"--disassemble={label}"]

        if not platform.nm_cmd:
            raise NmError(f"No nm command for {platform.id}")

        try:
            nm_proc = subprocess.run(
                " ".join([platform.nm_cmd] + [str((target_path))]), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
        except subprocess.TimeoutExpired as e:
            raise NmError("Timeout expired")
        except subprocess.CalledProcessError as e:
            raise NmError.from_process_error(e)

        if nm_proc.stdout:
            # e.g.
            # 00000000 T osEepromRead
            #          U osMemSize
            for line in nm_proc.stdout.splitlines():
                nm_line = line.split()
                if len(nm_line) == 3 and label == nm_line[2]:
                    start_addr = int(nm_line[0], 16)
                    return [f"--start-address={start_addr}"]

        return ["--start-address=0"]

    @staticmethod
    def parse_objdump_flags(diff_flags: List[str]) -> List[str]:
        known_objdump_flags = ["-Mreg-names=32", "-Mno-aliases"]
        ret = []

        for flag in known_objdump_flags:
            if flag in diff_flags:
                ret.append(flag)

        return ret

    @staticmethod
    def run_objdump(
        target_data: bytes,
        platform: Platform,
        config: asmdiffer.Config,
        label: str,
        flags: List[str],
    ) -> str:
        flags = [flag for flag in flags if not flag.startswith(ASMDIFF_FLAG_PREFIX)]
        flags += [
            "--disassemble",
            "--disassemble-zeroes",
            "--line-numbers",
            "--reloc",
        ]
        
        sandbox_temp_dir = TemporaryDirectory()
        sandbox_path = Path(sandbox_temp_dir.name)
        if True:
            target_path = sandbox_path / "target.o"
            target_path.write_bytes(target_data)

            flags += DiffWrapper.get_objdump_target_function_flags(
                sandbox_path, target_path, platform, label
            )

            flags += config.arch.arch_flags

            if platform.objdump_cmd:
                try:
                    return subprocess.run(
                        " ".join(platform.objdump_cmd.split()
                        + flags
                        + [str(target_path)]),
                        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                    ).stdout.decode("utf-8")
                except subprocess.TimeoutExpired as e:
                    raise ObjdumpError("Timeout expired")
                except subprocess.CalledProcessError as e:
                    raise ObjdumpError.from_process_error(e)
            else:
                raise ObjdumpError(f"No objdump command for {platform.id}")

        return out

    @staticmethod
    def get_dump(
        elf_object: bytes,
        platform: Platform,
        diff_label: str,
        config: asmdiffer.Config,
        diff_flags: List[str],
    ) -> str:
        if len(elf_object) == 0:
            raise AssemblyError("Asm empty")

        try:
            basedump = DiffWrapper.run_objdump(
                elf_object, platform, config, diff_label, diff_flags
            )
        except ObjdumpError as e:
            print(e)
        if not basedump:
            raise ObjdumpError("Error running objdump")

        # Preprocess the dump
        try:
            basedump = asmdiffer.preprocess_objdump_out(
                None, elf_object, basedump, config
            )
        except AssertionError as e:
            logger.exception("Error preprocessing dump")
            raise DiffError(f"Error preprocessing dump: {e}")
        except Exception as e:
            raise DiffError(f"Error preprocessing dump: {e}")

        return basedump

    @staticmethod
    def diff(
        target_o : bytes,
        platform: Platform,
        diff_label: str,
        compiled_elf: bytes,
        diff_flags: List[str],
    ) -> DiffResult:

        try:
            arch = asmdiffer.get_arch(platform.arch or "")
        except ValueError:
            logger.error(f"Unsupported arch: {platform.arch}. Continuing assuming mips")
            arch = asmdiffer.get_arch("mips")

        objdump_flags = DiffWrapper.parse_objdump_flags(diff_flags)

        config = DiffWrapper.create_config(arch, diff_flags)

        basedump = DiffWrapper.get_dump(
            target_o,
            platform,
            diff_label,
            config,
            objdump_flags,
        )
        try:
            mydump = DiffWrapper.get_dump(
                compiled_elf, platform, diff_label, config, objdump_flags
            )
        except Exception as e:
            mydump = ""

        try:
            display = asmdiffer.Display(basedump, mydump, config)
        except Exception as e:
            raise DiffError(f"Error running asm-differ: {e}")

        try:
            # TODO: It would be nice to get a python object from `run_diff()` to avoid the
            # JSON roundtrip. See https://github.com/simonlindholm/asm-differ/issues/56
            result = json.loads(display.run_diff()[0])
            result["error"] = None
        except Exception as e:
            raise DiffError(f"Error running asm-differ: {e}")

        return result

def make_pkl():
    pickle.dump(scrape_scratches(), open("training.pkl", "wb"))

def download_compilers():
    download_n64()
    download_ps1()

def compile_scratch(diff_label, ctx_c, code_c, compiler_id, compiler_flags) -> CompilationResult:
    compiler = compiler_from_id(compiler_id)
    return CompilerWrapper.compile_code(compiler, compiler_flags, code_c, ctx_c, diff_label)

def diff_compilation(diff_label, platform, target_o, compilation: CompilationResult) -> DiffResult:
    return DiffWrapper.diff(
        target_o,
        platform_from_id(platform),
        diff_label,
        bytes(compilation.elf_object),
        diff_flags=[],
    )

def update_score(prev_score, diff: DiffResult) -> None:
    score = diff.get("current_score", prev_score)
    diff["reward"] = prev_score - score
    diff["score"] = score
    return diff

def compile_and_update(prev_score, diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o, initial=False):
    score = prev_score
    try:
        compilation = compile_scratch(diff_label, ctx_c, code_c, compiler, compiler_flags)
    except CompilationError as e:
        print(e)
        if initial:
            raise e
        return {
            "score": 10000,
            "reward": prev_score - 10000,
        }
    
    diff = diff_compilation(diff_label, platform, target_o, compilation)
    return update_score(prev_score, diff)

def load_pkl():
    return pickle.load(open("training.pkl", "rb"))

import numpy as np
from gym import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import clang
import pycparser
import decomp_permuter.src.perm.perm as perm
import decomp_permuter.src.perm.parse as parse
import decomp_permuter.src.perm.eval as eval
import decomp_permuter.src.perm.ast as ast
import decomp_permuter.src.permuter as permuter
import decomp_permuter.src.compiler as perm_compiler
import decomp_permuter.src.candidate as candidate
import decomp_permuter.src.helpers as perm_helpers


class DecompilationEnv(gym.Env):
    ACTION_PERM_TEMP_FOR_EXPR = 0
    ACTION_PERM_EXPAND_EXPR = 1
    ACTION_PERM_CONDITION = 2
    ACTION_PERM_REMOVE_AST = 3
    ACTION_PERM_REFER_TO_VAR = 4
    ACTION_PERM_STRUCT_REF = 5
    ACTION_PERM_REORDER_DECLS = 6
    ACTION_PERM_INLINE = 7
    ACTION_PERM_ADD_MASK = 8
    ACTION_PERM_COMMUTATIVE = 9
    ACTION_PERM_CAST_SIMPLE = 10
    ACTION_PERM_RANDOMIZE_FUNCTION_TYPE = 11
    ACTION_PERM_MAX = 12



    def __init__(self, training_data): # training_data is a tuple list of (diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o)
        super(DecompilationEnv, self).__init__()
        self.training_data = training_data
        self.action_space = spaces.Discrete(DecompilationEnv.ACTION_PERM_MAX)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.current_code = random.randint(0, len(training_data) - 1)
        self.code_state : dict[int, dict[str, Any]] = {}
        self.initial_score = 0
        self.best_score = self.initial_score
        self.n_steps_since_last_reset = 0
        self.n_steps = 0

    def step(self, action):
        print('action', action)
        diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o = self.training_data[self.current_code]
        # permute the code
        if self.current_code in self.code_state:
            prev_score = self.code_state[self.current_code]["prev_score"]
            ctx_c = ""
            code_c = self.code_state[self.current_code]["last_permutation"].replace("#pragma _permuter randomizer start", "").replace("#pragma _permuter randomizer end", "").lstrip()
        else:
            self.code_state[self.current_code] = {}
            base_compilation = compile_and_update(0, diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o)
            if base_compilation['reward'] == -1: # compilation failed
                return np.array([0]), 0, True, False, {}
            prev_score = base_compilation["score"]
            self.initial_score = prev_score
            code_c = ctx_c + "\n\n" + code_c
        with open("code.c", "w") as f:
            f.write(code_c)
            try:
                code_c = pycparser.preprocess_file("code.c", cpp_path="cpp", cpp_args=["-E", "-P", "-w"])
            except:
                return np.array([prev_score]), 0, True, False, {}
        code_perm = parse.perm_parse(code_c)
        eval_state = perm.EvalState()
        try:
            cand = candidate.Candidate.from_source(code_perm.evaluate(random.randint(0, code_perm.perm_count), eval_state), eval_state, diff_label,{
    "perm_temp_for_expr": 100 if action == DecompilationEnv.ACTION_PERM_TEMP_FOR_EXPR else 0,
    "perm_expand_expr": 100 if action == DecompilationEnv.ACTION_PERM_EXPAND_EXPR else 0,
    "perm_reorder_stmts": 0,
    "perm_reorder_decls": 100 if action == DecompilationEnv.ACTION_PERM_REORDER_DECLS else 0,
    "perm_add_mask": 100 if action == DecompilationEnv.ACTION_PERM_ADD_MASK else 0,
    "perm_xor_zero": 0,
    "perm_cast_simple": 100 if action == DecompilationEnv.ACTION_PERM_CAST_SIMPLE else 0,
    "perm_refer_to_var": 100 if action == DecompilationEnv.ACTION_PERM_REFER_TO_VAR else 0,
    "perm_float_literal": 0,
    "perm_randomize_internal_type": 0,
    "perm_randomize_external_type": 0,
    "perm_randomize_function_type": 100 if action == DecompilationEnv.ACTION_PERM_RANDOMIZE_FUNCTION_TYPE else 0,
    "perm_split_assignment": 0,
    "perm_sameline": 1,
    "perm_ins_block": 0,
    "perm_struct_ref": 100 if action == DecompilationEnv.ACTION_PERM_STRUCT_REF else 0,
    "perm_empty_stmt": 0,
    "perm_condition": 100 if action == DecompilationEnv.ACTION_PERM_CONDITION else 0,
    "perm_mult_zero": 0,
    "perm_dummy_comma_expr": 0,
    "perm_add_self_assignment": 0,
    "perm_commutative": 100 if action == DecompilationEnv.ACTION_PERM_COMMUTATIVE else 0,
    "perm_add_sub": 0,
    "perm_inequalities": 0,
    "perm_compound_assignment": 0,
    "perm_remove_ast": 100 if action == DecompilationEnv.ACTION_PERM_REMOVE_AST else 0,
    "perm_duplicate_assignment": 0,
    "perm_chain_assignment": 0, 
    "perm_long_chain_assignment": 0,
    "perm_pad_var_decl": 0,
    "perm_inline": 100 if action == DecompilationEnv.ACTION_PERM_INLINE else 0,
            },random.randint(0, 1000000))
        except:
            return np.array([prev_score]).astype(np.float32), 0, True, False, {}
        cand.randomize_ast()
        permutation : str = cand.get_source()
        diff_result = compile_and_update(prev_score, diff_label, platform, "", permutation, target_s, compiler, compiler_flags, target_o)
        json.dump(diff_result, open('tmp.json','w'),indent=4)
        self.code_state[self.current_code]["prev_score"] = diff_result["score"]
        self.code_state[self.current_code]["strength"] = self.initial_score - diff_result["score"]
        self.code_state[self.current_code]["last_permutation"] = permutation if (diff_result["score"] < prev_score and diff_result["score"] < self.best_score) else code_c
        self.code_state[self.current_code]["cur_permutation"] = permutation
        reward = diff_result["reward"] + (self.code_state[self.current_code]["strength"] - self.strength_total) * 0.1
        diff_result["last_permutation"] = permutation
        open("tmp.c", 'w').write(permutation)
        self.n_steps_since_last_reset += 1
        if diff_result["score"] < self.best_score:
            self.best_score = diff_result["score"]
        self.n_steps += 1
        return np.array([diff_result["score"]]).astype(np.float32), reward, (diff_result["score"] <= (self.initial_score / 3) and diff_result['score'] >= 0) or self.n_steps_since_last_reset == 1000, False, diff_result

    def render(self, mode='console'):
        pass

    def close(self):
        pass

    def reset(self):
        if self.n_steps > 0:
            if "strength" not in self.code_state[self.current_code]:
                #self.strength_total = 0
                pass
            else:
                self.strength_total += self.code_state[self.current_code]["strength"]
        else:
            self.strength_total = 0
        self.current_code = random.randint(0, len(self.training_data) - 1)
        diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o = self.training_data[self.current_code]
        if len(target_o) > 2048:
            return self.reset()
        self.n_steps_since_last_reset = 0
        if ("extern ? " in code_c):
            return self.reset()
        try:
            self.initial_score = compile_and_update(0, diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o, True)["score"]
        except:
            return self.reset()

        self.best_score = self.initial_score
        return np.array([self.initial_score]).astype(np.float32), None

from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, env: DummyVecEnv, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env
    
    def _on_step(self) -> bool:
        env = self.env.envs[0]
        try:
            self.logger.record("cur_permutation", env.code_state[env.current_code]["cur_permutation"])
            self.logger.record("score", env.code_state[env.current_code]["prev_score"])
            self.logger.record("strength", env.strength_total + env.code_state[env.current_code]["strength"]) # positive=good, negative=bad
            self.logger.dump(step=self.num_timesteps)
        except:
            pass
        # save model every 1000 steps
        if self.num_timesteps % 1000 == 0:
            self.model.save("ppo_decomp")
        return True
    



def test_env():
    env = DummyVecEnv([lambda: DecompilationEnv(load_pkl())])

    obs = env.reset()

    model = PPO('MlpPolicy', env, verbose=1,
                tensorboard_log="./ppo_decomp_tensorboard/",
                
    ).learn(50000, callback=TensorboardCallback(env))
    model.save("ppo_decomp")
    env.render()

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        print ("Step {}".format(step + 1))
        obs, reward, done, info = env.step(action)
        print('obs=', obs, 'reward=', reward, 'done=', done)
        env.render()
        if done:
            print("Function matching")
            obs = env.reset()

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, choices=['train', 'make_pkl', 'download_compilers'])
    args = parser.parse_args()
    if args.command == 'train':
        test_env()
    elif args.command == 'make_pkl':
        make_pkl()
    elif args.command == 'download_compilers':
        download_compilers()
    else:
        parser.print_help()

    
