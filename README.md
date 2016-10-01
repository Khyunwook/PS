#BFS
```c++
	queue<int> q; //matrix
	check[1] = true; q.push(1);
	while(!q.empty()){
		int x = q.front(); q.pop();
		printf("%d", x);
		for(int i=1; i<=n; i++){
			if(a[x][i] == 1 && check[i] == false){
				check[i] = true;
				q.push(i);
			}
		}
	}
//----------------------------------------------------------------------------
	queue<int> q; //list
	check[1] = true; q.push(1);
	while(!q.empty()){
		int x = q.front(); q.pop();
		printf("%d", x);
		for(int i=1; i<=a[x].size(); i++){
			int y = a[x][i];
			if( check[i] == false){
				check[i] = true;
				q.push(y);
			}
		}
	}

	
```

#DFS
```c++
void dfs(int x){ //matrix
	check[x] = true;
	printf("%d ",x);
	for(int i=1; i<=n; i++){
		if(a[x][i] == 1 && check[i] == false){
			dfs(i);
		}
	}
}

void dfs(int x){ //list
	check[x] = true;
	printf("%d ",x);
	for(int i=0; i<a[x].size(); i++){
		int y = a[x][i];
		if(check[y] == false){
			dfs(y);
		}
	}
}
```

#최장 공통 부분 수열(LCS)
```c++
int LCS(string& input, string& compare){
    int cache[1001][1001];
    memset(cache,0,sizeof(cache));
    for(int i=1;i<=compare.length();i++){
        for(int j=1; j<=input.length();j++){
            if(compare[i-1] == input[j-1]){
                cache[i][j] = cache[i-1][j-1] +1;
                }
            else{
                cache[i][j] = max(cache[i-1][j], cache[i][j-1]);
            }
        }
    }
    return cache[compare.length()][input.length()];
}
```
#LCS 백트래킹
```
string output;
void backTracking(int m, int n){
    if(m==0 || n ==0) return;
    if(cache[m][n] > cache[m-1][n-1] && cache[m][n] > cache[m][n-1] && cache[m][n] > cache[m-1][n]){
        //문자열 인덱스는 캐시 인덱스보다 1씩 더 작다. 
        output = input[n-1] + output;
        backTracking(m-1, n-1);
    }else if(cache[m][n] > cache[m-1][n] && cache[m][n] == cache[m][n-1]){
        backTracking(m, n-1);
    }else{
          backTracking(m-1, n);
    }
}
```
#세그먼트 트리

```c++
// a: 배열 a
// tree: 세그먼트 트리
// node: 세그먼트 트리 노드 번호
// node가 담당하는 합의 범위가 start ~ end
long long init(vector<long long> &a, vector<long long> &tree, int node, int start, int end) {
    if (start == end) {
        return tree[node] = a[start];
    } else {
        return tree[node] = init(a, tree, node*2, start, (start+end)/2) + init(a, tree, node*2+1, (start+end)/2+1, end);
    }
}
// node가 담당하는 구간이 start~end이고, 구해야하는 합의 범위는 left~right
void update(vector<long long> &tree, int node, int start, int end, int index, long long diff) {
    if (index < start || index > end) return;
    tree[node] = tree[node] + diff;
    if (start != end) {
        update(tree,node*2, start, (start+end)/2, index, diff);
        update(tree,node*2+1, (start+end)/2+1, end, index, diff);
    }
}
long long sum(vector<long long> &tree, int node, int start, int end, int left, int right) {
    if (left > end || right < start) {
        return 0;
    }
    if (left <= start && end <= right) {
        return tree[node];
    }
    return sum(tree, node*2, start, (start+end)/2, left, right) + sum(tree, node*2+1, (start+end)/2+1, end, left, right);
}
```

#팬윅트리
```c++
#include <cstdio>
#include <vector>
using namespace std;
long long sum(vector<long long> &tree, int i) {
    long long ans = 0;
    while (i > 0) {
        ans += tree[i];
        i -= (i & -i);
    }
    return ans;
}
void update(vector<long long> &tree, int i, long long diff) {
    while (i < tree.size()) {
        tree[i] += diff;
        i += (i & -i);
    }
}
```

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

#MST PRIM
```c++
for(int i=0;i<line;i++){
        int a,b,c;
        scanf("%d %d %d" , &a, &b, &c);
        arr[a].push_back(make_pair(b,c));
        arr[b].push_back(make_pair(a,c));
    }
    
    check[1] = true;
    priority_queue<pair<int,int>> q;
    
    for (auto &p : arr[1]) {            //arr[1]에 있는 모든 pair형을 다 반복한다.
        q.push(make_pair(-p.second,p.first));
    }
    
    int answer =0;
    
    while(q.empty() != 1){
        auto p = q.top();               //auto로 인하여 p가 자동적으로 Pair형이 된다.
        q.pop();
        if(check[p.second] == true){    //끝점이 방문된적이 있다면 그 경우는 스킵한다.
            continue;
        }
        check[p.second] = true;
        answer += -p.first;
        int x = p.second;
        for (auto &p : arr[x]) {         //연결되는 모든 점의 간선들을 다 넣어준다.
            q.push(make_pair(-p.second,p.first));
        }
    }
    
    printf("%d\n" , answer);
```
#TSP
```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;
int n;
//가중치를 저장하기 위핸 배열 
int dist[15][15];
int TSP(vector<int> path, vector<bool> visited, int len){
    //모든 도시 다 방문했을 경우 
    if(path.size() == n) return len+dist[path.back()][0];
    int ret = 987654321;
    
    for(int next=0; next<n;next++){
        //방문 했다면 패스 
        if(visited[next]==true) continue;
        
        int cur = path.back();
        path.push_back(next);
        visited[next] = true;
        ret = min(ret,TSP(path,visited,len+dist[cur][next]));
        visited[next] = false; 
        path.pop_back();
    }
    return ret; 
}
int main(){
        cin >> n;
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                cin >> dist[i][j];
            }
        }
        vector<int> path(1, 0); // 경로를 저장할 벡터, 시작 도시 0번도시 선택. 
        vector<bool> visited(n, false); // 방문 여부를 저장할 벡터. false로 초기화. 
        visited[0] = true; // 출발 도시 방문여부 체크.
        double ret = TSP(path, visited, 0);
        cout << ret << endl;
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
#소인수분해
```c++
for(int i=2; i*i <=n; i++){
	while(n%i == 0){
		printf("%d\n",i);
		n/=i;
	}
}
if(n>1){
	printf("%d\n",n);
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

#에라토스테네스의 체(소수 빨리구하기)
```c++
void Eratos(int n)
{
	// 배열을 동적할당 할 포인터 선언
	bool* PrimeArray;
	// 만일 n이 1보다 작거나 같으면 함수 종료
	if(n<=1) return;
	/*	2부터 n까지 n-1개를 저장할 수 있는 배열 할당
		배열 참조 번호와 소수와 일치하도록 배열의 크기는
		n+1 길이만큼 할당(인덱스 번호 0과 1은 사용하지 않음)	*/
	PrimeArray=new bool[n+1];
	/*	배열초기화
		처음엔 모두 소수로 보고 true값을 줌	*/
	for(int i=2; i<=n; i++) PrimeArray[i]=true;
	/*	에라토스테네스의 체에 맞게 소수를 구함
		만일, PrimeArray[i]가 true이면 i 이후의 i 배수는
		약수로 i를 가지고 있는 것이 되므로 i 이후의 i 배수에 대해
		false값을 준다.
		PrimeArray[i]가 false이면 i는 이미 소수가 아니므로 i의 배수 역시
		소수가 아니게 된다. 그러므로 검사할 필요도 없다.	*/
		//i*2 (정수 over flow 막기 위해)
	for(int i=2; (i*i)<=n; i++)
	{
		if(PrimeArray[i])
		{
			for(int j=i*i; j<=n; j+=i) PrimeArray[j]=false;
		}
	}
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

#DP 파일합치기
```c++
//D[i][j] = (i번째 수부터 j번째 수의 합) + min(D[i][k] + D[k+1][j]) for i ≤ k < j
#include <bits/stdc++.h>
using namespace std;
 
int T, N;
int A[501], S[501], D[501][501];
 
int dy(int s, int e)
{
    if (s >= e) return 0;
    if (s+1 == e) return A[s] + A[e];
    int &ret = D[s][e];
    if (ret < 2e9) return ret;
    for (int k=s;k<e;k++){
        ret = min(ret, dy(s, k) + dy(k+1, e) + S[e] - S[s-1]);
    }
    return ret;
}
 
int main()
{
    for (scanf("%d", &T);T--;){
        scanf("%d", &N);
        for (int i=1;i<=N;i++) scanf("%d", A+i), S[i] = S[i-1] + A[i];
        for (int i=1;i<=N;i++) for (int j=i;j<=N;j++) D[i][j] = 2e9;
        printf("%d\n", dy(1, N));
    }
}
```

#DP 카드게임
```c++
//D[i][j] = (i번째 카드부터 j번째 카드까지 적힌 점수 합) - min(D[i+1][j], D[i][j-1])
#include <bits/stdc++.h>
using namespace std;
 
int T, N;
int A[1003], S[1003];
int D[1003][1003];
 
int dy(int s, int e)
{
    if (s > e) return 0;
    if (s == e) return A[s];
    int &ret = D[s][e];
    if (ret >= 0) return ret;
    ret = max(S[e] - S[s-1] - dy(s+1, e), S[e] - S[s-1] - dy(s, e-1));
    return ret;
}
 
int main()
{
    for (scanf("%d", &T);T--;){
        scanf("%d", &N);
        for (int i=1;i<=N;i++) scanf("%d", A+i), S[i] = S[i-1] + A[i];
        for (int i=1;i<=N;i++) for (int j=i;j<=N;j++) D[i][j] = -1;
        printf("%d\n", dy(1, N));
    }
}
```
