//pass
//--local_size=[4,4] --num_groups=[10,10] --warp-sync=16

#define SIZE 4
#define TILES 10
#define LENGTH (TILES * SIZE)

__kernel void matrix_transpose(global float* A)
{
  local float tile [SIZE][SIZE];
  
  int x = get_local_id(0);
  int y = get_local_id(1);

  int tile_x = get_group_id(0);
  int tile_y = get_group_id(1);

	tile[x][y] = A[((x + (tile_x * SIZE)) * LENGTH) + (y + (tile_y * SIZE))];

	tile[x][y] = tile[y][x];

	barrier(CLK_GLOBAL_MEM_FENCE);

	A[((x + (tile_y * SIZE)) * LENGTH) + (y + (tile_x * SIZE))] = tile[x][y];
}
