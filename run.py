"""为整理后的核心实现提供统一入口，所有路径均相对当前副本解析。"""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
PATHS = [ROOT/'StreamingThinker/evaluate', ROOT/'StreamingThinker', ROOT/'Streaming_EDU/src', ROOT/'EDU_codex/src', ROOT]
sys.path[:0] = [str(p) for p in PATHS]
ENTRIES = {
    'segment': ('module', 'final_edu.cli'),
    'train-segmenter': ('module', 'stream_edu.train_variant'),
    'eval-segmenter': ('module', 'stream_edu.eval_seq'),
    'prepare-inputs': ('file', 'StreamingThinker/scripts/prepare_full_edu_inputs.py'),
    'sft': ('file', 'StreamingThinker/finetune_streaming_strict_modelonly_best.py'),
    'evaluate': ('file', 'StreamingThinker/evaluate/evaluate_streaming_metrics_online_edu.py'),
    'rl': ('module', 'ST_RL.EDU_HRPO.train_edu_hrpo_fsdp'),
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in {'-h', '--help'}:
        print('Usage: python run.py <command> [arguments]\nCommands: ' + ', '.join(ENTRIES))
        return
    command = sys.argv[1]
    if command not in ENTRIES:
        raise SystemExit('Unknown command: ' + command)
    kind, target = ENTRIES[command]
    sys.argv = [target, *sys.argv[2:]]
    if kind == 'module':
        runpy.run_module(target, run_name='__main__')
    else:
        runpy.run_path(str(ROOT/target), run_name='__main__')

if __name__ == '__main__':
    main()
