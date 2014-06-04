#include <cstdlib>

int main() {
  for(int i = 0; i < 5; i++) {
    if(i == 0)
      system("./mand > cpuC++Bench.txt");
    else
      system("./mand >> cpuC++Bench.txt");
  }
  return(0);
}
