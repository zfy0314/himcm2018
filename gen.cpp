#include <iostream>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>
#include <algorithm>

#define MAXN 10005
#define ri register int
#define int ll

typedef long long ll;

const int N = 5000;

using namespace std;

int curt[MAXN], newt[MAXN];
int curh[MAXN], newh[MAXN];
int cura[MAXN], newa[MAXN];
int curp[MAXN], newp[MAXN];
int people[MAXN];
int sum = 0;

inline int read_int()
{
	register int ret = 0, f = 1; register char c = getchar();
	while(c < '0' || c > '9') {if(c == '-') f = -1; c = getchar();}
	while(c >= '0' && c <= '9') {ret = (ret << 1) + (ret << 3) + int(c - 48); c = getchar();}
	return ret * f;
}

inline void file()
{
	freopen("H.csv", "r", stdin);
	freopen("testdata.in", "w", stdout);
}

signed main()
{
	//Vancouver
	file();
	srand(time(NULL));
	default_random_engine e1, e2;
	normal_distribution <double> H(75, 5);
	normal_distribution <double> T(24, 0.3);
	for(ri i = 1; i <= N; i++)
	{
		cin >> curh[i];
		if(curh[i] < 70)
			newh[i] = 70;
		else
			newh[i] = 59;
	}
	fclose(stdin);
	freopen("T.csv", "r", stdin);
	for(ri i = 1; i <= N; i++)
	{
		double tmp;
		cin >> tmp;
		tmp -= 272.15;
		curt[i] = int(tmp + 0.5);
		sum += curt[i];
	}
	ri cnt = 0;
	for(ri i = 1; i <= N; i++)
	{
		if(curt[i] * N < sum)
			newt[i] = 30;
		else
			newt[i] = 20;
	}
	fclose(stdin);
	default_random_engine e3, e4;
	normal_distribution <double> S(50, 10);
	normal_distribution <double> L(300, 20);
	for(ri i = 1; i <= N; i++)
	{
		ri opt = rand() % 20;
		if(opt == 0)
			curp[i] = int(L(e3) + 0.5);
		else
			curp[i] = int(S(e4) + 0.5);
		newp[i] = curp[i] >= 100 ? 1 : 0;
	}
	for(ri i = 1; i <= N; i++)
		cura[i] = rand() % 5 == 0 ? 3 : rand() % 2, newa[i] = cura[i] >= 2 ? 1 : 0;
	puts("4 4");
	for(ri i = 1; i <= N; i++)
		printf("%lld %lld %lld %lld %lld %lld %lld %lld\n", curt[i], curh[i], cura[i], curp[i], newt[i], newh[i], newa[i], newp[i]);
	return 0;
}
