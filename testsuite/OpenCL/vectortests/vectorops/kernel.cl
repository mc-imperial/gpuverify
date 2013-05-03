//pass
//--local_size=64 --num_groups=64



__kernel void foo() {
	
	int4 splat = 1;
	__assert(splat.x == 1 & splat.y == 1 & splat. z == 1 & splat.w == 1);
	
	int4 res = (int4)(5,5,5,5);
	int4 v = (int4)(1,2,3,4);
	
	res.wy = v.zy;
	__assert(res.x == 5 & res.y == 2 & res.z == 5 & res.w == 3);
	
	res = (int4)(5,5,5,5);
	
	int4 ad = (int4)(0,1,2,0);
	
	res += ad;
	__assert(res.x == 5 & res.y == 6 & res.z == 7 & res.w == 5);
	
	res = (int4)(5,5,5,5);
	
	res.xyzw += ad.yzwx;
	
	__assert(res.x == 6 & res.y == 7 & res.z == 5 & res.w == 5);
	
	res = (int4)(5,5,5,5);
	
	res = -res;
	
	__assert(res.x == -5 & res.y == -5 & res.z == -5 & res.w == -5);
	
}


