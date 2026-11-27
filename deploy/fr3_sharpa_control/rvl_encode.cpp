// Wilson RVL nibble stream. The caller adds ROS compressedDepth header/dimensions.
#include <cstdint>
#include <cstddef>
extern "C" size_t sharpa_rvl(const uint16_t* data, size_t n, uint32_t* out) {
 size_t words=0; uint32_t word=0; unsigned used=0;
 auto nibble=[&](uint32_t v){word=(word<<4)|v;if(++used==8){out[words++]=word;word=0;used=0;}};
 auto vle=[&](uint32_t v){while(v>7){nibble((v&7)|8);v>>=3;}nibble(v);};
 size_t i=0;int32_t previous=0;
 while(i<n){size_t start=i;while(i<n && data[i]==0)++i;vle(i-start);start=i;while(i<n && data[i]!=0)++i;vle(i-start);
  for(size_t j=start;j<i;++j){int32_t delta=int32_t(data[j])-previous;previous=data[j];vle(delta>=0?uint32_t(delta)*2:uint32_t(-delta)*2-1);}}
 if(used)out[words++]=word<<(4*(8-used));return words*4;
}
