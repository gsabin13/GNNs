dt=$(date | awk '{print$2$3"-"$4"-"$5$6}')
echo "Date: " $dt
env_output=$dt"_env_info.txt"
prog_output=$dt"_running_log.txt"
echo $env_output
echo $prog_output