#include <map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef _MSC_VER
#include <stdint.h>
#include <direct.h>
#define FORMAT_SIZET "Iu"
#define stdrup _strdup
#define tempnam _tempnam
#else
#include <unistd.h>
#define FORMAT_SIZET "zu"
#endif

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


using namespace std;

struct file_line_time {
	string file;
	int line;
	time_t time;
};

struct arg_data {
	size_t size;
	void* data;
	struct file_line_time flt;
};

struct kernel_data {
	cl_program program;
	std::string name;
	std::vector<struct arg_data> args;
	unsigned int dimension;
	size_t global_size [3];
	size_t local_size [3];
};

void de_newline(char* string) {
	for (int i = 0; string[i] != '\0'; i++) {
		if (string[i] == '\n' || string[i] == '\r') {
			string[i] = ' ';
		}
	}
}

class CL_Logger {
public:
	std::map<cl_program, pair<vector<string>,struct file_line_time> > programs;
	std::map<cl_program, string> options;
	std::map<cl_kernel, struct kernel_data> kernels;
	const char* dirname;

	void dump(cl_kernel karnol, const char* file, int line) {
		struct kernel_data kernel = kernels[karnol];

		FILE* f = fopen(tempnam(dirname,kernel.name.c_str()),"w");
			
		fprintf(f,"// --local_size=%" FORMAT_SIZET, kernel.local_size[0]);
		if (kernel.dimension > 1) {
			for (unsigned int i = 1; i < kernel.dimension; i++) {
				fprintf(f,",%" FORMAT_SIZET,kernel.local_size[i]);
			}
		}

		fprintf(f," --global_size=%" FORMAT_SIZET,kernel.global_size[0]);
		if (kernel.dimension > 1) {
			for (unsigned int i = 1; i < kernel.dimension; i++) {
				fprintf(f,",%" FORMAT_SIZET,kernel.global_size[i]);
			}
		}

		fprintf(f," --kernel-args=%s",kernel.name.c_str());
		for (unsigned int i = 0; i < kernel.args.size(); i++) {
			cl_kernel_arg_address_qualifier q;
			clGetKernelArgInfo(karnol, i, CL_KERNEL_ARG_ADDRESS_QUALIFIER, sizeof(q), &q, NULL);
			if (q == CL_KERNEL_ARG_ADDRESS_PRIVATE) { // Best heuristic for "scalar"
				switch (kernel.args[i].size) {
				case 1: fprintf(f, ",%hhu", *(uint8_t*)  kernel.args[i].data); break;
				case 2: fprintf(f, ",%hu",  *(uint16_t*) kernel.args[i].data); break;
				case 4: fprintf(f, ",%u",   *(uint32_t*) kernel.args[i].data); break;
				case 8: fprintf(f, ",%lu",  *(uint64_t*) kernel.args[i].data); break;
				default:
					fprintf(f, ",0x");
					for (int j = kernel.args[i].size -1; j >= 0; j--) {
						fprintf(f, "%02X", ((unsigned char*) kernel.args[i].data)[j]);
					}
					break;
				}
			}
		}

		char* opts = strdup(options[kernel.program].c_str());
		de_newline(opts);
		fprintf(f, " %s\n", opts);
		fprintf(f, "// Built at %s:%d\n",programs[kernel.program].second.file.c_str(), programs[kernel.program].second.line);
		fprintf(f, "// Run at %s:%d\n", file, line);

		fprintf(f, "\n");

		std::vector<std::string> code = programs[kernel.program].first;
		for (size_t i = 0; i < code.size(); i++) {
			fprintf(f, "%s", code[i].c_str());
		}
		fclose(f);
	}

	CL_Logger (void) {
		dirname = getenv("GPUV_KI_DIR") ? getenv("GPUV_KI_DIR") : ".gpuverify";

		// Making our directory in case it isn't there already
		// TODO: Does this work on Windows?
		struct stat st = {0};
		if (stat(dirname, &st) == -1) {
#ifdef _MSC_VER
      _mkdir(dirname);
#else
			mkdir(dirname,0700);
#endif
		}

	}

};

CL_Logger& singleton(void)
{
	static CL_Logger t;
	return t;
}

extern "C" {

	cl_program clCreateProgramWithSource_hook (cl_context context,
						   cl_uint count,
						   const char **strings,
						   const size_t *lengths,
						   cl_int *errcode_ret,
						   const char* file, int line)
	{
		cl_program program = clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
		std::vector<std::string> code (count);
		if (lengths == NULL) {
			for (cl_uint i = 0; i < count; i++) {
				std::string line (strings[i]);
				code.push_back(line);
			}
		}
		else {
			for (cl_uint i = 0; i < count; i++) {
				if (lengths[i] == 0) {
					std::string line (strings[i]);
					code.push_back(line);
				}
				else {
					std::string line (strings[i],lengths[i]);
					code.push_back(line);
				}
			}
		}

		struct file_line_time flt = {string(file), line, 0};
		singleton().programs[program] = pair <vector<string>,struct file_line_time> (code,flt);

		return program;
	}

	cl_int clBuildProgram_hook (cl_program program,
				    cl_uint num_devices,
				    const cl_device_id *device_list,
				    const char *options,
				    void (CL_CALLBACK *  pfn_notify)(cl_program, void *),
				    void *user_data)
	{
		std::string opts (options ? options : "");
		singleton().options[program] = opts;
		return clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);
	}

	cl_kernel clCreateKernel_hook (cl_program  program,
				       const char *kernel_name,
				       cl_int *errcode_ret)
	{
		cl_kernel kernel = clCreateKernel(program, kernel_name, errcode_ret);
		struct kernel_data data;
		data.program = program;
		data.name = std::string (kernel_name);
		singleton().kernels[kernel] = data;
		return kernel;
	}

	cl_int clSetKernelArg_hook (cl_kernel kernel,
				    cl_uint arg_index,
				    size_t arg_size,
				    const void *arg_value)
	{
		cl_int ret = clSetKernelArg(kernel, arg_index, arg_size, arg_value);

		// Resizing the vector if need-be
		if (singleton().kernels[kernel].args.size() < arg_index+1) {
			singleton().kernels[kernel].args.resize(arg_index + 1);
		}

		singleton().kernels[kernel].args[arg_index].size = arg_size;
		if (arg_value) {
			singleton().kernels[kernel].args[arg_index].data = malloc(arg_size);
			memcpy(singleton().kernels[kernel].args[arg_index].data, arg_value, arg_size);
		}
		else {
			singleton().kernels[kernel].args[arg_index].data = NULL;
		}
		return ret;
	}

	cl_int clEnqueueNDRangeKernel_hook (cl_command_queue command_queue,
					    cl_kernel kernel,
					    cl_uint work_dim,
					    const size_t *global_work_offset,
					    const size_t *global_work_size,
					    const size_t *local_work_size,
					    cl_uint num_events_in_wait_list,
					    const cl_event *event_wait_list,
					    cl_event *event,
					    const char* file, int line)

	{
		cl_int ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
		singleton().kernels[kernel].dimension = work_dim;
		for (cl_uint i = 0; i < work_dim; i++) {
			singleton().kernels[kernel].global_size[i] = global_work_size[i];
			singleton().kernels[kernel].local_size[i] = local_work_size[i];
		}
		singleton().dump(kernel, file, line);
		return ret;
	}

}
