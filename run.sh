

for ((i=0;i<20;i++)); do
    python runNNet3_6000.py
done
python process_trace.py
mv cost.txt cost_[6000]_16_16_16.txt
mv grad.txt grad_[6000]_16_16_16.txt
mv param.txt param_[6000]_16_16_16.txt
rm trace*.pk

for ((i=0;i<20;i++)); do
    python runNNet3_600.py
done
python process_trace.py
mv cost.txt cost_[600]_16_16_16.txt
mv grad.txt grad_[600]_16_16_16.txt
mv param.txt param_[600]_16_16_16.txt
rm trace*.pk

for ((i=0;i<20;i++)); do
    python runNNet3_120.py
done
python process_trace.py
mv cost.txt cost_[120]_16_16_16.txt
mv grad.txt grad_[120]_16_16_16.txt
mv param.txt param_[120]_16_16_16.txt
rm trace*.pk