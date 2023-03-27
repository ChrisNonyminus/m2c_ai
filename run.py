import hashlib
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


def scrape_scratches(url = "https://decomp.me/api/scratch", scratches_list : list[tuple] = [], recursion_depth = 500):
    scratches = jsonpickle.decode(requests.get(url).text)
    next_url = scratches['next']
    for result in scratches["results"]:
        if result['platform'] in ['n64', 'ps1'] and (result['score'] >= 300 and result['score'] < 1000) and "gcc" in result['compiler']:
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

def get_scratch(slug):
    scratch = jsonpickle.decode(requests.get("https://decomp.me/api/scratch" + "/" + slug).text)
    with open("zips/" + scratch['slug'] + ".zip", 'wb') as f:
        f.write(requests.get("https://decomp.me/api/scratch" + "/" + scratch['slug'] + "/export").content)
    with zipfile.ZipFile("zips/" + scratch['slug'] + ".zip", mode="r") as archive:
        ctx_c = archive.read("ctx.c").decode("utf-8") if "ctx.c" in archive.namelist() else ""
        code_c = archive.read('code.c').decode("utf-8")
        target_s = archive.read('target.s').decode('utf-8')
        target_o = archive.read('target.o')
        diff_label = scratch['name']
        platform = scratch['platform']
        compiler = scratch['compiler']
        compiler_flags = jsonpickle.decode(requests.get("https://decomp.me/api/scratch" + "/" + scratch['slug']).text)['compiler_flags']
        return (diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o)

def get_most_recent_slug(url = "https://decomp.me/api/scratch"):
    scratches = jsonpickle.decode(requests.get(url).text)
    next_url = scratches['next']
    for result in scratches["results"]:
        if result['platform'] in ['n64', 'ps1'] and (result['score'] >= 300 and result['score'] < 1000) and "gcc" in result['compiler']:
            return result['slug']
    return get_most_recent_slug(next_url)

from diff_wrapper import DiffWrapper

def prepare_dataset():
    existing = load_pkl() if os.path.exists("training.pkl") else []
    pickle.dump(scrape_scratches("https://decomp.me/api/scratch", existing), open("training.pkl", "wb"))

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
        else:
            return {
                "score": 10000,
                "reward": prev_score - 10000,
            }
    
    diff = diff_compilation(diff_label, platform, target_o, compilation)
    return update_score(prev_score, diff)

def display_diff(diff: DiffResult) -> None:
    out_file = open("diff.txt", "w")
    for row in diff["rows"]:
        if "base" in row:
            out_file.write(" ".join([text["text"] for text in row["base"]['text']]) + " " * (64 - len(" ".join([text["text"] for text in row["base"]['text']]))))
        else:
            out_file.write(" " * 64)
        if "current" in row:
            out_file.write(" ".join([text["text"] for text in row["current"]['text']]))
        out_file.write("\n")

def load_pkl():
    return pickle.load(open("training.pkl", "rb"))

import numpy as np
from gym import spaces
from stable_baselines3 import PPO, A2C
import clang, clang.cindex
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
    ACTION_PERM_REORDER_STMTS = 8
    ACTION_PERM_COMMUTATIVE = 9
    ACTION_PERM_CAST_SIMPLE = 10
    ACTION_PERM_RANDOMIZE_FUNCTION_TYPE = 11
    ACTION_PERM_MAX = 12



    def __init__(self, training_data): # training_data is a tuple list of (diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o)
        super(DecompilationEnv, self).__init__()
        self.training_data = training_data
        self.action_space = spaces.Discrete(DecompilationEnv.ACTION_PERM_MAX)
        # values we need to observe:
        # - current score
        # - current permutation (source code)
        # base (row["base"]["text"]) vs current (row["base"]["text"]) for each row in diff_result["rows"]
        self.observation_space = spaces.Dict({
            "score": spaces.Box(low=0, high=100000, shape=(1,), dtype=np.float32),
            "code": spaces.Box(low=0, high=100000, shape=(131072,), dtype=np.float32),
            "diff": spaces.Box(low=0, high=100000, shape=(8192,), dtype=np.float32),
            "target_asm": spaces.Box(low=0, high=100000, shape=(131072,), dtype=np.float32),
            "current_asm": spaces.Box(low=0, high=100000, shape=(131072,), dtype=np.float32),
        })
        self.current_code = random.randint(0, len(training_data) - 1)
        self.code_state : dict[int, dict[str, Any]] = {}
        self.initial_score = 0
        self.best_score = self.initial_score
        self.n_steps_since_last_reset = 0
        self.n_steps = 0
        self.scratch : tuple = None

    def step(self, action):
        print('action', action)
        diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o = self.training_data[self.current_code] if self.scratch is None else self.scratch
        # permute the code
        if self.current_code in self.code_state:
            prev_score = self.code_state[self.current_code]["prev_score"]
            ctx_c = ""
            code_c = self.code_state[self.current_code]["last_permutation"].replace("#pragma _permuter randomizer start", "").replace("#pragma _permuter randomizer end", "").lstrip()
        else:
            self.code_state[self.current_code] = {}
            base_compilation = compile_and_update(0, diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o)
            if base_compilation['reward'] == -1: # compilation failed
                return {}, 0, True, False, {}
            prev_score = base_compilation["score"]
            self.initial_score = prev_score
            code_c = ctx_c + "\n\n" + code_c
        with open("code.c", "w") as f:
            f.write(code_c)
            try:
                code_c = pycparser.preprocess_file("code.c", cpp_path="cpp", cpp_args=["-E", "-P", "-w"])
            except:
                return {}, 0, True, False, {}
        code_perm = parse.perm_parse(code_c)
        eval_state = perm.EvalState()
        try:
            cand = candidate.Candidate.from_source(code_perm.evaluate(random.randint(0, code_perm.perm_count), eval_state), eval_state, diff_label,{
    "perm_temp_for_expr": 100 if action == DecompilationEnv.ACTION_PERM_TEMP_FOR_EXPR else 0,
    "perm_expand_expr": 100 if action == DecompilationEnv.ACTION_PERM_EXPAND_EXPR else 0,
    "perm_reorder_stmts": 100 if action == DecompilationEnv.ACTION_PERM_REORDER_STMTS else 0,
    "perm_reorder_decls": 100 if action == DecompilationEnv.ACTION_PERM_REORDER_DECLS else 0,
    "perm_add_mask": 0,
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
            cand.randomize_ast()
        except:
            return {}, 0, True, False, {}
        permutation : str = cand.get_source()
        diff_result = compile_and_update(prev_score, diff_label, platform, "", permutation, target_s, compiler, compiler_flags, target_o)
        json.dump(diff_result, open('tmp.json','w'),indent=4)
        self.code_state[self.current_code]["prev_score"] = diff_result["score"]
        self.code_state[self.current_code]["strength"] = self.initial_score - diff_result["score"]
        self.code_state[self.current_code]["last_permutation"] = permutation if (diff_result["score"] < prev_score and diff_result["score"] <= self.best_score) else code_c
        self.code_state[self.current_code]["cur_permutation"] = permutation
        reward = diff_result["reward"] + (self.code_state[self.current_code]["strength"] - self.strength_total) * 0.1
        if diff_result["score"] <= 100:
            reward += 1000
            self.code_state[self.current_code]["strength"] = 1000
        diff_result["last_permutation"] = permutation
        open("tmp.c", 'w').write(permutation)
        # clang-format tmp.c
        subprocess.run(["clang-format", "tmp.c", "-i"])
        self.n_steps_since_last_reset += 1
        if diff_result["score"] < self.best_score:
            self.best_score = diff_result["score"]
        if "rows" in diff_result:
            display_diff(diff_result)
        self.n_steps += 1
        diff_rows = diff_result["rows"] if "rows" in diff_result else [
            {"base": {"text": [{"text": ""}]}},
            {"current": {"text": [{"text": "CANNOT COMPILE"}]}},
        ]
        for row in diff_rows:
            if "base" not in row:
                row["base"] = {"text": [{"text": ""}]}
            if "current" not in row:
                row["current"] = {"text": [{"text": ""}]}
        diff_base_rows = [" ".join([text["text"] for text in row["base"]['text']]) for row in diff_rows]
        diff_current_rows = [" ".join([text["text"] for text in row["current"]['text']]) for row in diff_rows]
        diff_rows = list((diff_base_rows, diff_current_rows))
        diff_rows_hash_diff_per_row = [hashlib.sha256((row[0].strip().encode("utf-8"))) == hashlib.sha256((row[1].strip().encode("utf-8"))) for row in diff_rows]
        # resize the permutation so that when we convert it to a numpy array it is the same shape as (131072,)
        permutation = permutation + " " * (131072 - len(permutation))
        permutation = permutation[:131072]
        # do the same for the diff_rows_hash_diff_per_row
        diff_rows_hash_diff_per_row = diff_rows_hash_diff_per_row + [True] * (8192 - len(diff_rows_hash_diff_per_row))

        target_asm_str = "\n".join(diff_base_rows)  + " " * (131072 - len("\n".join(diff_base_rows)))
        target_asm_str = target_asm_str[:131072]
        target_asm = np.fromstring(target_asm_str, dtype=np.uint8)

        current_asm_str = "\n".join(diff_current_rows) + " " * (131072 - len("\n".join(diff_current_rows)))
        current_asm_str = current_asm_str[:131072]
        current_asm = np.fromstring(current_asm_str, dtype=np.uint8)
        diff_result["strength"] = self.strength_total + self.code_state[self.current_code]["strength"]
        return {
            "score": np.array([diff_result["score"]]),
            "code": np.fromstring(permutation, dtype=np.uint8),
            "diff": np.fromiter(diff_rows_hash_diff_per_row, dtype=np.bool),
            "target_asm": target_asm,
            "current_asm": current_asm,

        }, reward, (diff_result["score"] <= (self.initial_score / 4) and diff_result['score'] >= 0) or self.n_steps_since_last_reset == 1000 or (diff_result["score"] <= 100 and diff_result['score'] >= 0), False, diff_result

    def render(self, mode='console'):
        pass

    def close(self):
        pass

    def reset(self, scratch : tuple = None):
        if scratch is None:
            if self.n_steps > 0:
                if self.current_code not in self.code_state or "strength" not in self.code_state[self.current_code]:
                    #self.strength_total = 0
                    pass
                else:
                    self.strength_total += self.code_state[self.current_code]["strength"]
            else:
                self.strength_total = 0
            self.code_state.pop(self.current_code, None)
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
            diff_rows = compile_and_update(0, diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o, True)["rows"]
            for row in diff_rows:
                if "base" not in row:
                    row["base"] = {"text": [{"text": ""}]}
                if "current" not in row:
                    row["current"] = {"text": [{"text": ""}]}
            diff_base_rows = [" ".join([text["text"] for text in row["base"]['text']]) for row in diff_rows]
            diff_current_rows = [" ".join([text["text"] for text in row["current"]['text']]) for row in diff_rows]
            diff_rows = list((diff_base_rows, diff_current_rows))
            diff_rows_hash_diff_per_row = [hashlib.sha256((row[0].split(":")[1].strip().encode("utf-8"))) == hashlib.sha256((row[1].split(":")[1].strip().encode("utf-8"))) for row in diff_rows]
            # resize the permutation so that when we convert it to a numpy array it is the same shape as (131072,)
            code_c = code_c + " " * (131072 - len(code_c))
            # do the same for the diff_rows_hash_diff_per_row
            diff_rows_hash_diff_per_row = diff_rows_hash_diff_per_row + [True] * (8192 - len(diff_rows_hash_diff_per_row))
            return {
                "score": np.array([self.initial_score]),
                "code": np.fromstring(code_c, dtype=np.uint8),
                "diff": np.fromiter(diff_rows_hash_diff_per_row, dtype=np.bool),
                "target_asm": np.fromstring("\n".join(diff_base_rows) + " " * (131072 - len("\n".join(diff_base_rows))), dtype=np.uint8),
                "current_asm": np.fromstring("\n".join(diff_current_rows) + " " * (131072 - len("\n".join(diff_current_rows))), dtype=np.uint8),
            }, {
                "strength": self.strength_total,
            }
        else:
            self.scratch = scratch
            self.current_code = 0
            diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o = scratch
            self.initial_score = compile_and_update(0, diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o, True)["score"]
            self.best_score = self.initial_score
            diff_rows = compile_and_update(0, diff_label, platform, ctx_c, code_c, target_s, compiler, compiler_flags, target_o, True)["rows"]
            for row in diff_rows:
                if "base" not in row:
                    row["base"] = {"text": [{"text": ""}]}
                if "current" not in row:
                    row["current"] = {"text": [{"text": ""}]}
            diff_base_rows = [" ".join([text["text"] for text in row["base"]['text']]) for row in diff_rows]
            diff_current_rows = [" ".join([text["text"] for text in row["current"]['text']]) for row in diff_rows]
            diff_rows = list((diff_base_rows, diff_current_rows))
            diff_rows_hash_diff_per_row = [hashlib.sha256((row[0].split(":")[1].strip().encode("utf-8"))) == hashlib.sha256((row[1].split(":")[1].strip().encode("utf-8"))) for row in diff_rows]
            # resize the permutation so that when we convert it to a numpy array it is the same shape as (131072,)
            code_c = code_c + " " * (131072 - len(code_c))
            # do the same for the diff_rows_hash_diff_per_row
            diff_rows_hash_diff_per_row = diff_rows_hash_diff_per_row + [True] * (8192 - len(diff_rows_hash_diff_per_row))
            return {
                "score": np.array([self.initial_score]),
                "code": np.fromstring(code_c, dtype=np.uint8),
                "diff": np.fromiter(diff_rows_hash_diff_per_row, dtype=np.bool),
                "target_asm": np.fromstring("\n".join(diff_base_rows) + " " * (131072 - len("\n".join(diff_base_rows))), dtype=np.uint8),
                "current_asm": np.fromstring("\n".join(diff_current_rows) + " " * (131072 - len("\n".join(diff_current_rows))), dtype=np.uint8),
            }, {
                "strength": self.strength_total,
            }


from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, env, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env
    
    def _on_step(self) -> bool:
        env = self.env
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
    env = DecompilationEnv(load_pkl())

    obs, _ = env.reset()

    model = PPO('MultiInputPolicy', env, verbose=1,
                tensorboard_log="./ppo_decomp_tensorboard/",
                
    ).learn(50000, callback=TensorboardCallback(env))
    model.save("ppo_decomp")

def continue_training():
    env = DecompilationEnv(load_pkl())

    obs, _ = env.reset()

    model = PPO.load("ppo_decomp", env=env, verbose=1,
                tensorboard_log="./ppo_decomp_tensorboard/",
                
    ).learn(50000, callback=TensorboardCallback(env))
    model.save("ppo_decomp")

def run_on_scratch(scratch : tuple, do_next_if_done : bool = False):
    env = DecompilationEnv(load_pkl())
    env.scratch = scratch
    obs, _ = env.reset(scratch)
    model = PPO.load("ppo_decomp", env=env, verbose=1)
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, trunc, info = env.step(action)
        if info["score"] <= 100:
            if do_next_if_done:
                obs, _ = env.reset(get_scratch(get_most_recent_slug()))
            else:
                break
    return env.code_state[env.current_code]["cur_permutation"]


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, choices=['train', 'prepare_dataset', 'download_compilers', 'run'])
    parser.add_argument('--scratch', type=str, default=None)
    parser.add_argument('--do_next_if_done', type=bool, default=False)
    args = parser.parse_args()
    if args.command == 'train':
        if os.path.exists("ppo_decomp.zip"):
            continue_training()
        else:
            test_env()
    elif args.command == 'prepare_dataset':
        prepare_dataset()
    elif args.command == 'download_compilers':
        download_compilers()
    elif args.command == 'run':
        if os.path.exists("ppo_decomp.zip"):
            if args.scratch is not None:
                print(run_on_scratch(get_scratch(args.scratch), args.do_next_if_done))
            else:
                (run_on_scratch(get_scratch(get_most_recent_slug()), args.do_next_if_done))
        else:
            print("No model found. Please train first.")
    else:
        parser.print_help()

    
