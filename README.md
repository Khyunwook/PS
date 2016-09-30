#가장 가까운 두점 찾기
```c++
#include <cstdio>
#include <vector>
#include <set>
#include <algorithm>
using namespace std;
struct Point {
    int x, y;
    Point() {
    }
    Point(int x, int y) : x(x), y(y) {
    }
    bool operator < (const Point &v) const {
        if (y == v.y) {
            return x < v.x;
        } else {
            return y < v.y;
        }
    }
};
bool cmp(const Point &u, const Point &v) {
    return u.x < v.x;
}
int dist(Point p1, Point p2) {
    return (p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y);
}
int main() {
    int n;
    scanf("%d",&n);
    vector<Point> a(n);
    for (int i=0; i<n; i++) {
        scanf("%d %d",&a[i].x,&a[i].y);
    }
    sort(a.begin(), a.end(), cmp);
    set<Point> candidate = {a[0], a[1]};
    int ans = dist(a[0], a[1]);
    int start = 0;
    for (int i=2; i<n; i++) {
        Point now = a[i];
        while (start < i) {
            auto p = a[start];
            int x = now.x - p.x;
            if (x*x > ans) {
                candidate.erase(p);
                start += 1;
            } else {
                break;
            }
        }
        int d = (int)sqrt((double)ans)+1;
        auto lower_point = Point(-100000, now.y-d);
        auto upper_point = Point(100000, now.y+d);
        auto lower = candidate.lower_bound(lower_point);
        auto upper = candidate.upper_bound(upper_point);
        for (auto it = lower; it != upper; it++) {
            int d = dist(now, *it);
            if (d < ans) {
                ans = d;
            }
        }
        candidate.insert(now);
    }
    printf("%d\n",ans);
    //두점의 거리의 제곱을 출력
    return 0;
}
```

#DP 1로 만들기
```c++
int solv(int N){
     
    dp[1]=0;
    for(int i=2; i<=N; i++){
        dp[i]=dp[i-1]+1;
        if(i%3==0 && dp[i]>dp[i/3]+1)
            dp[i] = dp[i/3]+1;
        if(i%2==0 && dp[i]>dp[i/2]+1)
            dp[i] = dp[i/2]+1;
    }
     
    return dp[N];
}
```
#Floyd
```c++
for (int k = 0; k < N; ++k)
    {
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                if (d[i][j] > d[i][k] + d[k][j])
                {
                    d[i][j] = d[i][k] + d[k][j];
                    via[i][j] = k;
                }
            }
        }
    }
```

#대각선 채우기(행렬의 곱샘)
```c++
for (i = 2; i <= N; i++)//범위
         for (j = 1; j + i - 1 <= N; j++) //1~N
            for (k = 1; k <= i; k++)//
            {
               A[j][j + i - 1] = min(A[j][j + i - 1], A[j][j+k-1] + A[j+k][j + i - 1]);
            }
```
#반올림
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
