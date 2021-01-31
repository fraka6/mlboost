#!/bin/bash
# load data  
# usage: ./load.sh db table [year test]
# example: ./load.sh trial test 201501/01.tsv  (test)
#          ./load.sh trial stage2 201508/*.tsv (load full month full index)
#          ./load.sh trial stage2 201508/*.tsv --basic (load full month min index)
#          ./load.sh year stage2 "2015 --year" --basic (load full year min index)
#          ./load.sh year stage2 "2015 --year" --basic "--sock=/work/mysql2/mysql.sock"
db=${1:-'trial'}
table=${2:-'stage2'}
constraint=${3:-'2015'}
tsv_opt=${4:-''} # tsv2table options
mysql_opt=${5:-"--sock=/work/mysql/mysql.sock"} 

echo "#1) create new db "$db
cmd='echo "create database '$db';" | mysql -u root '$mysql_opt
echo $cmd
eval $cmd

echo "#2) create table "$table 
cmd="python tsv2table.py --db $db -t $table $tsv_opt | mysql -u root $mysql_opt"
echo $cmd
eval $cmd

echo "#3) load some data"
cmd="python loadtsv.py --db $db -t $table -c $constraint --mysql_opt='$mysql_opt'"
echo $cmd
eval $cmd
