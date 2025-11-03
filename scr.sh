rm tmp1 all_dchem.txt data.txt
for i in {0..9};do
cat fitness_${i}.txt >> fitness.txt
done
