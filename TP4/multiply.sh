if [ $# -eq 0 ]; then  echo "No arguments supplied";exit;fi

output=new_$1
rm -rf $output

multiplicacion=$2

AE_std=0.91
LE_std=0.33
AT_std=1.70
LT_std=1.28
AA_std=1.10
LA_std=0.67
NV_std=0.31

feature_std=()
feature_std[2]=$AE_std
feature_std[3]=$LE_std
feature_std[4]=$AT_std
feature_std[5]=$LT_std
feature_std[6]=$AA_std
feature_std[7]=$LA_std
feature_std[8]=$NV_std
c=1
while IFS= read -r line;do
	echo "$line"
	cl=1
	if [ $c -eq 1 ];then
		c=0
		echo "$line">>$output
		continue
	else
		IFS=',' read -r -a array <<< "$line" 
		declare -p array;
		for m in $(seq 1 $multiplicacion);do
			nueva_linea=""
			for i in "${!array[@]}";do
				factor=0
				if [ $i -ne 0 ] && [ $i -ne 1 ] && [ $i -ne 9 ] && [ $i -ne 10 ];then
					# echo -n "${array[$i]}   "
					# echo -n "${feature_std[$i]}   "
					while [ $factor -eq 0 ];do factor=$(seq -1 1 | shuf -n 1) ;done
					agregado=$(bc <<< "scale=4; ${feature_std[$i]} * $RANDOM * $factor / 32767")
					nuevo_numero=$(echo "${array[$i]} + ($agregado * 0.1 )" | bc -l)
					a_rounded=`printf "%.2f" $nuevo_numero`
					nueva_linea+="$a_rounded,"
				else
					if [ $i -eq 10 ];then 
						nueva_linea+="${array[$i]}"
					else
						nueva_linea+="${array[$i]},"
					fi
				fi
			done
		echo "$nueva_linea"
		echo "$nueva_linea">>$output
		done
	fi
	echo "----------"
done < "$1"
echo -n "$output  "
cat "$output"|wc -l



