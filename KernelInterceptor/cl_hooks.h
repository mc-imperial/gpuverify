#ifndef CL_HOOKS_H
#define CL_HOOKS_H

#ifdef __cplusplus
extern "C" {
#endif

cl_program clCreateProgramWithSource_hook (cl_context context,
                                           cl_uint count,
                                           const char **strings,
                                           const size_t *lengths,
                                           cl_int *errcode_ret,
					   const char* file, int line);

cl_int clBuildProgram_hook (cl_program program,
                            cl_uint num_devices,
                            const cl_device_id *device_list,
                            const char *options,
                            void (*pfn_notify)(cl_program, void *user_data),
                            void *user_data);

cl_kernel clCreateKernel_hook (cl_program  program,
                               const char *kernel_name,
                               cl_int *errcode_ret);

cl_int clSetKernelArg_hook (cl_kernel kernel,
                            cl_uint arg_index,
                            size_t arg_size,
                            const void *arg_value);

cl_int clEnqueueNDRangeKernel_hook (cl_command_queue command_queue,
                                    cl_kernel kernel,
                                    cl_uint work_dim,
                                    const size_t *global_work_offset,
                                    const size_t *global_work_size,
                                    const size_t *local_work_size,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event *event_wait_list,
                                    cl_event *event,
				    const char* file, int line);

#define clCreateProgramWithSource(a,b,c,d,e) clCreateProgramWithSource_hook(a,b,c,d,e,__FILE__,__LINE__)
#define clBuildProgram clBuildProgram_hook
#define clCreateKernel clCreateKernel_hook
#define clSetKernelArg clSetKernelArg_hook
#define clEnqueueNDRangeKernel(a,b,c,d,e,f,g,h,i) clEnqueueNDRangeKernel_hook(a,b,c,d,e,f,g,h,i,__FILE__,__LINE__)

#ifdef __cplusplus
}
#endif

#endif
