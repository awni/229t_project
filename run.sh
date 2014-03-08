# for ((i=0;i<001;i++)); do
#     python runNNet0.py
# done
# python process_trace.py
# mv cost.txt cost_.txt
# mv grad.txt grad_.txt
# mv param.txt param_.txt
# rm trace*.pk

# for ((i=0;i<001;i++)); do
#     python runNNet.py
# done
# python process_trace.py
# mv cost.txt cost_16.txt
# mv grad.txt grad_16.txt
# mv param.txt param_16.txt
# rm trace*.pk

# for ((i=0;i<001;i++)); do
#     python runNNet2.py
# done
# python process_trace.py
# mv cost.txt cost_16_16.txt
# mv grad.txt grad_16_16.txt
# mv param.txt param_16_16.txt
# rm trace*.pk

# for ((i=0;i<001;i++)); do
#     python runNNet3.py
# done
# python process_trace.py
# mv cost.txt cost_16_16_16.txt
# mv grad.txt grad_16_16_16.txt
# mv param.txt param_16_16_16.txt
# rm trace*.pk

# for ((i=0;i<001;i++)); do
#     python runNNet4.py
# done
# python process_trace.py
# mv cost.txt cost_16_16_16_16.txt
# mv grad.txt grad_16_16_16_16.txt
# mv param.txt param_16_16_16_16.txt
# rm trace*.pk

# for ((i=0;i<001;i++)); do
#     python runNNet5.py
# done
# python process_trace.py
# mv cost.txt cost_16_16_16_16_16.txt
# mv grad.txt grad_16_16_16_16_16.txt
# mv param.txt param_16_16_16_16_16.txt
# rm trace*.pk


# for ((i=0;i<001;i++)); do
#     python runNNet3_6000.py
# done
# python process_trace.py
# mv cost.txt cost_[6000]_16_16_16.txt
# mv grad.txt grad_[6000]_16_16_16.txt
# mv param.txt param_[6000]_16_16_16.txt
# rm trace*.pk

# for ((i=0;i<001;i++)); do
#     python runNNet3_600.py
# done
# python process_trace.py
# mv cost.txt cost_[600]_16_16_16.txt
# mv grad.txt grad_[600]_16_16_16.txt
# mv param.txt param_[600]_16_16_16.txt
# rm trace*.pk

for ((i=0;i<001;i++)); do
    python runNNet3_120.py
done
python process_trace.py
mv cost.txt cost_[120]_16_16_16.txt
mv grad.txt grad_[120]_16_16_16.txt
mv param.txt param_[120]_16_16_16.txt
rm trace*.pk