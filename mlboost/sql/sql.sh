#!/bin/bash
# run sql query on mysql db 
# usage: ./sql.sh query, [db, host, user]
# example: ./sql.sh "select session,ans from stage2 limit 5" >session_ans.5.tsv
#          ./sql.sh "select count(Distinct ans),count(Distinct tsStep) from stage2" test
#          ./sql.sh "select count(*) from stage2 WHERE DATE(tsTime)='2015-08-03'" trial
#          ./sql.sh "select min(tsTime),max(tsTime) from stage2;"
db=${2:-'year'}
host=${3:-'localhost'}
user=${4:-'root'}
opt=${5:-"--sock=/work/mysql/mysql.sock"}
query=${1:-""}

cmd="mysql -u "$user" -h "$host" "$opt
if [ "$query" != "" ];then
    new=$cmd" -e \"use "$db";"$query";\""
    cmd=$new
fi
echo $cmd
eval $cmd
