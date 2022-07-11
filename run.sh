pref="/uufs/chpc.utah.edu/common/home/u1320844/GNN_logs/"
dt=$(date | awk '{print$2$3"-"$4"-"$5$6}')

echo "Date: " $dt

git_output=$pref$dt"_git_info.txt"
env_output=$pref$dt"_env_info.txt"
prog_output=$pref$dt"_running_log.txt"

echo $env_output
echo $prog_output

git show > $git_output
git diff > $git_output
env > $env_output
pip list > $env_output

cmd=$1
echo $cmd
git add .
git commit -m $dt
git push origin master
pushd ../GNN_logs
git add .
git commit -m $dt
git push origin master
popd
