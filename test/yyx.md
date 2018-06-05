```c++
#include<iostream>
#include<iomanip>
#include<cstdlib>
#include<ctime>
#include<cmath>     //包含开平方sqrt()函数的头文件
using namespace std;
int main()
{
    srand((unsigned)time(NULL));
    int k = 0;
    int n = 0, beans = 0;
    double pi;
    double x = 0, y = 0;

    cout << "请输入你要投的豆子数量：" << endl;
    cin >> n;
    beans = n;
    do 
    {
        x = rand() % 2 + 0 + (double)(rand() % 1000) / 1000.0;
        y = rand() % 2 + 0 + (double)(rand() % 1000) / 1000.0;

        if (0 <= x&&x <= 1 && 0 <= y&&y <= 1) 
        {
            if (sqrt(x*x + y*y) <= 1) 
            {
                ++k;
                --n;
            }
            else --n;
        }
    } while (n > 0);
    pi = 4.0 * k / beans;
    cout << "近似值 π≈" << setiosflags(ios::fixed) << setprecision(10) << pi << endl;
    system("pause");
    return 0;
}
```