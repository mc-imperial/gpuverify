//pass
//--local_size=64 --num_groups=64 --no-inline



__kernel 
void foo()
{
	uchar4 start = (uchar4)(1,2,3,4);
	uchar4 temp4;
	uchar3 temp3;
	uchar2 temp2;
	
	temp4 = start.yyxx;
	temp3 = start.zyx;
	temp2 = temp3.zx;

	
}

