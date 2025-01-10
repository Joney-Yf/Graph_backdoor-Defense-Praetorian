learning_rates=(0.001)  # lr 参数的不同取值
mask_rates=(0.1)        # mask_rate 参数的不同取值
layers=(3)                          # layer 参数的不同取值
k_hops=(0)                          # k_hop 参数的不同取值
seeds=(456)                               # seeds 参数的不同取值
epochs=(1000)                           # 训练 epochs
num_hiddens=(1024)
num_decoder_layers=(1)
attacks=(GTA)
trigger_sizes=(3)
vs_sizes=(10)
# 循环遍历所有参数组合
for seed in "${seeds[@]}"
do
  for mask_rate in "${mask_rates[@]}"
  do
    for layer in "${layers[@]}"
    do
      for k_hop in "${k_hops[@]}"
      do
        for lr in "${learning_rates[@]}"
        do
          for epoch in "${epochs[@]}"
          do
            for num_decoder_layer in "${num_decoder_layers[@]}"
            do
              for num_hidden in "${num_hiddens[@]}"
              do
                for attack in "${attacks[@]}"
                do
                for ts in "${trigger_sizes[@]}"
                  do
                    for vs in "${vs_sizes[@]}"
                    do
                      for backdoor_file in ./Pubmed/*${attack}*train*1*
                      do
                        # 运行 Python 脚本，替换对应的参数
                        python defense.py \
                          --dataset pubmed \
                          --encoder gat \
                          --decoder gat \
                          --seed $seed \
                          --device 1 \
                          --backdoor_train_file "$backdoor_file" \
                          --use_cfg \
                          --concat_hidden \
                          --save_model \
                          --lr $lr \
                          --mask_rate $mask_rate \
                          --num_layers $layer \
                          --k_hop $k_hop \
                          --max_epoch $epoch \
                          --num_hidden $num_hidden \
                          --num_decoder_layer $num_decoder_layer\
                          --test_model GCN \
                          --target_class 0 >> ${attack}_Pubmed_time_cost.log
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
