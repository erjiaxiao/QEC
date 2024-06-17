#!/bin/bash
for LSS2 in 64 # 4 8 16 32 64 # 8 16
do

for D in 5 7 9 
do
	
for LSS1 in 4 8 16 32 64 128 256 
do


for Q in 4 8 16 32 64 128 256 # 2 4 8 16 32
do

for T in 3
do

for R in 1
do

rm run.sbatch
cp l.sbatch run.sbatch
echo "#SBATCH -o ./data4/d${D}_q${Q}_${LSS1}_${LSS2}_t${T}_r${R}" >> run.sbatch
#echo module use /opt/applics >> run.sbatch
echo module load cuda/10.0 >> run.sbatch
#echo ./execu -d $D -t ${T} -s 0 -q ${Q} -r ${R} -1 $LSS1 -2 $LSS2 -w "~/cuda/data2/d${D}_q${Q}_${LSS1}_${LSS2}_t${T}_r${R}" >> run.sbatch 
echo ./execu -d $D -t ${T} -s 0 -q ${Q} -r ${R} -1 $LSS1 -2 $LSS2 >> run.sbatch
sbatch run.sbatch

done
done
done
done
done
done
