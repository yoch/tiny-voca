cd /home/user/tiny-voca/bench/llama_cpp_fft
V="Pw Pt Pc T N R H H_R H_R_Pc H_R_Pc_N I_R_Pc_N lean_nofuse lean mid lean_R full full_N C Cnofuse CnoP CnoN"
bash /tmp/tv/rb.sh $V
for v in $V; do LD_LIBRARY_PATH=/tmp/tv/bins/ref taskset -c 2 /tmp/tv/real ab /tmp/tv/bins/ref /tmp/tv/bins/$v 40 dumps/u5.bin dumps/u6.bin | sed "s|/tmp/tv/bins/||g"; done
for p in "lean mid" "mid full" "lean lean_R" "lean_R full" "full full_N" "full C" "lean_nofuse lean" "H_R_Pc_N full"; do set -- $p; LD_LIBRARY_PATH=/tmp/tv/bins/ref taskset -c 2 /tmp/tv/real ab /tmp/tv/bins/$1 /tmp/tv/bins/$2 40 dumps/u5.bin dumps/u6.bin | sed "s|/tmp/tv/bins/||g"; done
echo DONE
