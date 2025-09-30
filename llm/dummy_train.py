from nemo.collections import llm
import nemo_run as run

def llama3_8b():
    # 1) レシピを構成（実行はまだ行われない）
    #    ※ ローカル最小実行を想定して GPU=1 にしてあります。
    #       マシンに応じて num_gpus_per_node を増減してください。
    recipe = llm.llama3_8b.pretrain_recipe(
        name="llama3_8b_dummy10",
        dir="./checkpoints_llama3_8b_dummy10",
        num_nodes=1,
        num_gpus_per_node=8,   # 例: 1GPU実行（要メモリ）
    )

    # 2) 最小・短時間化のための上書き（Python属性として直接設定）
    #    - 10ステップだけ回す
    #    - 途中保存を無効化
    #    - シーケンス長/バッチを縮小（メモリ圧縮）
    recipe.trainer.max_steps = 10
    recipe.trainer.enable_checkpointing = False
    recipe.trainer.log_every_n_steps = 1

    # モデル/データ側（必要に応じて更に小さく）
    recipe.model.config.seq_length = 512
    recipe.data.seq_length = 512
    recipe.data.global_batch_size = 8         # 総バッチ（DP×MBの積）
    recipe.data.micro_batch_size = 1          # 1GPUあたりのマイクロバッチ

    # 3) 実行エグゼキュータ（ローカル）
    executor = run.LocalExecutor()

    # 4) 実行
    #    LocalExecutor はローカルワークステーションでの単一タスク実行向け。
    #    複数タスク・複数環境なら run.Experiment を使えます。
    run.run(
        recipe,
        executor=executor,
        name="llama3_8b_pretraining_dummy10_local",
    )

def qwen2_1p5b():
        # 1) レシピを構成（実行はまだ行われない）
    #    ※ ローカル最小実行を想定して GPU=1 にしてあります。
    #       マシンに応じて num_gpus_per_node を増減してください。
    recipe = llm.qwen2_1p5b.pretrain_recipe(
        name="qwen2_1p5b_dummy10",
        dir="./checkpoints_qwen2_1p5b_dummy10",
        num_nodes=1,
        num_gpus_per_node=8,   # 例: 1GPU実行（要メモリ）
    )

    recipe.trainer.max_steps = 10
    recipe.trainer.enable_checkpointing = False
    recipe.trainer.log_every_n_steps = 1

    recipe.model.config.seq_length = 512
    recipe.data.seq_length = 512
    recipe.data.global_batch_size = 8         # 総バッチ（DP×MBの積）
    recipe.data.micro_batch_size = 1          # 1GPUあたりのマイクロバッチ

    # 3) 実行エグゼキュータ（ローカル）
    executor = run.LocalExecutor()

    run.run(
        recipe,
        executor=executor,
        name="qwen2_1p5b_pretraining_dummy10_local",
    )



if __name__ == "__main__":
    qwen2_1p5b()
