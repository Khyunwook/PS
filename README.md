#LIS 최장증가수열
```c++
//최대 공약수
for (int i = 1; i <= n; i++){
	dp[i] = 1; //1개를 선정하는 것이 기저이므로 1로 초기화 
	for (int j = 1; j < i; j++){
		if (a[i] > a[j] && dp[i]<dp[j]+1){
			dp[i] = dp[j] + 1;
		}
	}
	ret = max(ret, dp[i]);
}
```
#최대공약수, 최소 공배수
```c++
//최대 공약수
int gcd(int a, int b)
{
    if (b == 0)
        return a;
    gcd(b, a%b);
}
 
//최대 공배수
int lcm(int a, int b)
{
    return (a*b) / gcd(a, b);
}
```

#FASTSUM
```c++
int fastSum(int n){
	if(n==1)return 1;
	if(n%2==1) return fastSum(n-1) +n;
	return 2*fastSum(n/2) + (n/2)*(n/2);
}
```
#n개의 연속된 숫자중 m개를 뽑는 경우의 수
```c++
void printPicked(vector<int> picked){
	vector<int>::iterator itr;
	for (itr = picked.begin(); itr < picked.end(); itr++){
		printf("%d ", *itr);
	}printf("\n");
}

void pick(int n, vector<int>& picked, int toPick){
	if (toPick == 0) { printPicked(picked); return; }
	int smallest = picked.empty() ? 0 : picked.back() +1;
	for (int next = smallest; next < n; ++next){
		picked.push_back(next);
		pick(n, picked, toPick - 1);
		picked.pop_back();
	}
}

int main(void){

	vector <int> pickarr;
	pick(5, pickarr, 3);

	return 0;
}
```
