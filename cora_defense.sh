learning_rates=(0.001)  
mask_rates=(0.1)       
layers=(2)                        
k_hops=(0)                          
seeds=(1)                             
epochs=(1500)                          
num_hiddens=(512)
num_decoder_layers=(1)
attacks=(DPGBA UGBA SBA GTA)
trigger_sizes=(3)
vs_sizes=(10)
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
                      for backdoor_file in ./Cora/*${attack}*train*
                      do
                        python defense.py \
                          --dataset cora \
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
                          --target_class 0 >> ${attack}_Cora_time_cost.log
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
