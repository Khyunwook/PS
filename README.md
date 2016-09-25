대각선 채우기(행렬의 곱샘)
```c++
for (i = 2; i <= N; i++)//범위
         for (j = 1; j + i - 1 <= N; j++) //1~N
            for (k = 1; k <= i; k++)//
            {
               A[j][j + i - 1] = min(A[j][j + i - 1], A[j][j+k-1] + A[j+k][j + i - 1]);
            }
```
반올림
```c++
#define banollim(x,dig) (floor((x)*pow(10,dig)+0.5)/pow(10,dig))

double Rounding( double x, int digit )
{
    //digit자리까지 반올림
    return ( floor( (x) * pow( float(10), digit ) + 0.5f ) / pow( float(10), digit ) );
}
```
#LIS 최장증가수열(O(n^2))
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
#LIS 최장증가수열(O(nlogn))
```c++
vector<int> LIS;
LIS.push_back(v[0]);
for(int i=1; i<v.size(); ++i){
    if(LIS.back()<v[i]) LIS.push_back(v[i]); //LIS의 마지막 값이 v[i]보다 작으면 추가한다.
    else{
        //LIS 에서 이진 탐색으로 v[i]보다 큰 수 중 가장 작은수와 교체 한다.
        vector<int>::iterator it=lower_bound(LIS.begin(), LIS.end(), v[i]);
        *it=v[i];
    }
}
cout<<LIS.size()<<endl;
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
