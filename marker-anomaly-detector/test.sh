MY_VAR=$(grep MY_VAR .env | xargs)
IFS='=' read -ra MY_VAR <<< "$MY_VAR"
MY_VAR=${MY_VAR[1]}
echo $MY_VAR