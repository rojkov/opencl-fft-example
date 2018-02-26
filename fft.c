#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "pgm.h"

#define PI 3.14159265358979

#define MAX_SOURCE_SIZE (0x100000)

#define AMP(a, b) (sqrt((a) * (a) + (b) * (b)))

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue queue = NULL;
cl_program program = NULL;

enum Mode { forward = 0, inverse = 1 };

int setWorkSize(size_t *gws, size_t *lws, cl_int x, cl_int y) {
        switch (y) {
        case 1:
                gws[0] = x;
                gws[1] = 1;
                lws[0] = 16;
                lws[1] = 16;
                break;
        default:
                gws[0] = x;
                gws[1] = y;
                lws[0] = 16;
                lws[1] = 16;
                break;
        }

        return 0;
}

int fftCore(cl_mem dst, cl_mem src, cl_mem spin, cl_int m,
            enum Mode direction) {
        cl_int ret;

        cl_int iter;
        cl_uint flag;

        cl_int n = 1 << m;

        cl_event kernelDone;

        cl_kernel brev = NULL;
        cl_kernel bfly = NULL;
        cl_kernel norm = NULL;

        cl_ulong start;
        cl_ulong end;

        brev = clCreateKernel(program, "bitReverse", &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create kernel. Error: %d\n", ret);
                exit(1);
        }
        bfly = clCreateKernel(program, "butterfly", &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create kernel. Error: %d\n", ret);
                exit(1);
        }
        norm = clCreateKernel(program, "norm", &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create kernel. Error: %d\n", ret);
                exit(1);
        }

        size_t gws[2];
        size_t lws[2];

        switch (direction) {
        case forward:
                flag = 0x00000000;
                break;
        case inverse:
                flag = 0x80000000;
                break;
        }

        ret = clSetKernelArg(brev, 0, sizeof(cl_mem), (void *)&dst);
        if (ret != 0) {
                fprintf(stderr, "Can't set kernel argument. Error: %d\n", ret);
                exit(1);
        }
        ret = clSetKernelArg(brev, 1, sizeof(cl_mem), (void *)&src);
        if (ret != 0) {
                fprintf(stderr, "Can't set kernel argument. Error: %d\n", ret);
                exit(1);
        }
        ret = clSetKernelArg(brev, 2, sizeof(cl_int), (void *)&m);
        if (ret != 0) {
                fprintf(stderr, "Can't set kernel argument. Error: %d\n", ret);
                exit(1);
        }
        ret = clSetKernelArg(brev, 3, sizeof(cl_int), (void *)&n);
        if (ret != 0) {
                fprintf(stderr, "Can't set kernel argument. Error: %d\n", ret);
                exit(1);
        }

        ret = clSetKernelArg(bfly, 0, sizeof(cl_mem), (void *)&dst);
        if (ret != 0) {
                fprintf(stderr, "Can't set kernel argument. Error: %d\n", ret);
                exit(1);
        }
        ret = clSetKernelArg(bfly, 1, sizeof(cl_mem), (void *)&spin);
        if (ret != 0) {
                fprintf(stderr, "Can't set kernel argument. Error: %d\n", ret);
                exit(1);
        }
        ret = clSetKernelArg(bfly, 2, sizeof(cl_int), (void *)&m);
        if (ret != 0) {
                fprintf(stderr, "Can't set kernel argument. Error: %d\n", ret);
                exit(1);
        }
        ret = clSetKernelArg(bfly, 3, sizeof(cl_int), (void *)&n);
        if (ret != 0) {
                fprintf(stderr, "Can't set kernel argument. Error: %d\n", ret);
                exit(1);
        }
        ret = clSetKernelArg(bfly, 5, sizeof(cl_uint), (void *)&flag);
        if (ret != 0) {
                fprintf(stderr, "Can't set kernel argument. Error: %d\n", ret);
                exit(1);
        }

        ret = clSetKernelArg(norm, 0, sizeof(cl_mem), (void *)&dst);
        if (ret != 0) {
                fprintf(stderr, "Can't set kernel argument. Error: %d\n", ret);
                exit(1);
        }
        ret = clSetKernelArg(norm, 1, sizeof(cl_int), (void *)&n);
        if (ret != 0) {
                fprintf(stderr, "Can't set kernel argument. Error: %d\n", ret);
                exit(1);
        }

        /* Reverse bit ordering */
        setWorkSize(gws, lws, n, n);
        ret = clEnqueueNDRangeKernel(queue, brev, 2, NULL, gws, lws, 0, NULL,
                                     NULL);
        if (ret != 0) {
                fprintf(stderr, "Can't enqueue kernel. Error: %d\n", ret);
                exit(1);
        }

        /* Perform Butterfly Operations*/
        setWorkSize(gws, lws, n / 2, n);
        printf("m: %d\nn: %d\ngws: [%ld, %ld]\nlws: [%ld, %ld]\n", m, n, gws[0],
               gws[1], lws[0], lws[1]);
        for (iter = 1; iter <= m; iter++) {
                ret = clSetKernelArg(bfly, 4, sizeof(cl_int), (void *)&iter);
                if (ret != 0) {
                        fprintf(stderr,
                                "Can't set kernel argument. Error: %d\n", ret);
                        exit(1);
                }
                ret = clEnqueueNDRangeKernel(queue, bfly, 2, NULL, gws, lws, 0,
                                             NULL, &kernelDone);
                if (ret != 0) {
                        fprintf(stderr, "Can't enqueue kernel. Error: %d\n",
                                ret);
                        exit(1);
                }
                ret = clWaitForEvents(1, &kernelDone);
                if (ret != 0) {
                        fprintf(stderr, "Failed to wait for event. Error: %d\n",
                                ret);
                        exit(1);
                }
                ret = clGetEventProfilingInfo(kernelDone,
                                              CL_PROFILING_COMMAND_START,
                                              sizeof(cl_ulong), &start, NULL);
                if (ret != 0) {
                        fprintf(stderr,
                                "Failed get profiling info. Error: %d\n", ret);
                        exit(1);
                }
                ret = clGetEventProfilingInfo(kernelDone,
                                              CL_PROFILING_COMMAND_END,
                                              sizeof(cl_ulong), &end, NULL);
                if (ret != 0) {
                        fprintf(stderr,
                                "Failed get profiling info. Error: %d\n", ret);
                        exit(1);
                }
                printf("Butterfly operation: %10.5f [ms]\n",
                       (end - start) / 1000000.0);
        }

        if (direction == inverse) {
                setWorkSize(gws, lws, n, n);
                ret = clEnqueueNDRangeKernel(queue, norm, 2, NULL, gws, lws, 0,
                                             NULL, &kernelDone);
                if (ret != 0) {
                        fprintf(stderr, "Can't enqueue kernel. Error: %d\n",
                                ret);
                        exit(1);
                }
                ret = clWaitForEvents(1, &kernelDone);
                if (ret != 0) {
                        fprintf(stderr, "Failed to wait for event. Error: %d\n",
                                ret);
                        exit(1);
                }
        }

        ret = clReleaseKernel(bfly);
        if (ret != 0) {
                fprintf(stderr, "Can't release kernel. Error: %d\n", ret);
                exit(1);
        }
        ret = clReleaseKernel(brev);
        if (ret != 0) {
                fprintf(stderr, "Can't release kernel. Error: %d\n", ret);
                exit(1);
        }
        ret = clReleaseKernel(norm);
        if (ret != 0) {
                fprintf(stderr, "Can't release kernel. Error: %d\n", ret);
                exit(1);
        }

        return 0;
}

int main() {
        cl_mem xmobj = NULL;
        cl_mem rmobj = NULL;
        cl_mem wmobj = NULL;
        cl_kernel sfac = NULL;
        cl_kernel trns = NULL;
        cl_kernel hpfl = NULL;

        cl_platform_id platform_id = NULL;

        cl_uint ret_num_devices;
        cl_uint ret_num_platforms;

        cl_int ret;

        cl_float2 *xm;
        cl_float2 *rm;
        cl_float2 *wm;

        pgm_t ipgm;
        pgm_t opgm;

        FILE *fp;
        const char fileName[] = "./fft.cl";
        size_t source_size;
        char *source_str;
        cl_int i, j;
        cl_int n;
        cl_int m;

        int r;

        size_t gws[2];
        size_t lws[2];

        /* Load kernel source code */
        fp = fopen(fileName, "r");
        if (!fp) {
                fprintf(stderr, "Failed to load kernel.\n");
                exit(1);
        }
        source_str = (char *)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);

        /* Read image */
        r = readPGM(&ipgm, "lena.pgm");
        if (r < 0) {
                fprintf(stderr, "Wrong input image format. Exiting...\n");
                exit(1);
        }

        n = ipgm.width;
        m = (cl_int)(log((double)n) / log(2.0));

        xm = (cl_float2 *)malloc(n * n * sizeof(cl_float2));
        rm = (cl_float2 *)malloc(n * n * sizeof(cl_float2));
        wm = (cl_float2 *)malloc(n / 2 * sizeof(cl_float2));

        for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                        ((float *)xm)[(2 * n * j) + 2 * i + 0] =
                            (float)ipgm.buf[n * j + i];
                        ((float *)xm)[(2 * n * j) + 2 * i + 1] = (float)0;
                }
        }

        /* Get platform/device  */
        ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
        if (ret != 0) {
                fprintf(stderr, "Can't get platform. Error: %d\n", ret);
                exit(1);
        }
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,
                             &ret_num_devices);
        if (ret != 0) {
                fprintf(stderr, "Can't get device id. Error: %d\n", ret);
                exit(1);
        }

        /* Create OpenCL context */
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create context. Error: %d\n", ret);
                exit(1);
        }

        /* Create Command queue */
        queue = clCreateCommandQueue(context, device_id,
                                     CL_QUEUE_PROFILING_ENABLE, &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create queue. Error: %d\n", ret);
                exit(1);
        }

        /* Create Buffer Objects */
        xmobj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               n * n * sizeof(cl_float2), NULL, &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create buffer. Error: %d\n", ret);
                exit(1);
        }
        rmobj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               n * n * sizeof(cl_float2), NULL, &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create buffer. Error: %d\n", ret);
                exit(1);
        }
        wmobj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               (n / 2) * sizeof(cl_float2), NULL, &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create buffer. Error: %d\n", ret);
                exit(1);
        }

        /* Transfer data to memory buffer */
        ret =
            clEnqueueWriteBuffer(queue, xmobj, CL_TRUE, 0,
                                 n * n * sizeof(cl_float2), xm, 0, NULL, NULL);
        if (ret != 0) {
                fprintf(stderr, "Can't write buffer. Error: %d\n", ret);
                exit(1);
        }

        /* Create kernel program from source */
        program =
            clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                      (const size_t *)&source_size, &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create program. Error: %d\n", ret);
                exit(1);
        }

        /* Build kernel program */
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if (ret != 0) {
                fprintf(stderr, "Can't build program. Error: %d\n", ret);
                exit(1);
        }

        /* Create OpenCL Kernel */
        sfac = clCreateKernel(program, "spinFact", &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create kernel. Error: %d\n", ret);
                exit(1);
        }
        trns = clCreateKernel(program, "transpose", &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create kernel. Error: %d\n", ret);
                exit(1);
        }
        hpfl = clCreateKernel(program, "highPassFilter", &ret);
        if (ret != 0) {
                fprintf(stderr, "Can't create kernel. Error: %d\n", ret);
                exit(1);
        }

        /* Create spin factor */
        ret = clSetKernelArg(sfac, 0, sizeof(cl_mem), (void *)&wmobj);
        ret = clSetKernelArg(sfac, 1, sizeof(cl_int), (void *)&n);
        setWorkSize(gws, lws, n / 2, 1);
        ret = clEnqueueNDRangeKernel(queue, sfac, 1, NULL, gws, lws, 0, NULL,
                                     NULL);
        if (ret != 0) {
                fprintf(stderr, "Can't enqueue kernel. Error: %d\n", ret);
                exit(1);
        }

        /* Butterfly Operation */
        fftCore(rmobj, xmobj, wmobj, m, forward);

        /* Transpose matrix */
        ret = clSetKernelArg(trns, 0, sizeof(cl_mem), (void *)&xmobj);
        ret = clSetKernelArg(trns, 1, sizeof(cl_mem), (void *)&rmobj);
        ret = clSetKernelArg(trns, 2, sizeof(cl_int), (void *)&n);
        setWorkSize(gws, lws, n, n);
        ret = clEnqueueNDRangeKernel(queue, trns, 2, NULL, gws, lws, 0, NULL,
                                     NULL);
        if (ret != 0) {
                fprintf(stderr, "Can't enqueue kernel. Error: %d\n", ret);
                exit(1);
        }

        /* Butterfly Operation */
        fftCore(rmobj, xmobj, wmobj, m, forward);

        /* Apply high-pass filter */
        cl_int radius = n / 8;
        ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&rmobj);
        ret = clSetKernelArg(hpfl, 1, sizeof(cl_int), (void *)&n);
        ret = clSetKernelArg(hpfl, 2, sizeof(cl_int), (void *)&radius);
        setWorkSize(gws, lws, n, n);
        ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL,
                                     NULL);
        if (ret != 0) {
                fprintf(stderr, "Can't enqueue kernel. Error: %d\n", ret);
                exit(1);
        }

        /* Inverse FFT */

        /* Butterfly Operation */
        fftCore(xmobj, rmobj, wmobj, m, inverse);

        /* Transpose matrix */
        ret = clSetKernelArg(trns, 0, sizeof(cl_mem), (void *)&rmobj);
        ret = clSetKernelArg(trns, 1, sizeof(cl_mem), (void *)&xmobj);
        setWorkSize(gws, lws, n, n);
        ret = clEnqueueNDRangeKernel(queue, trns, 2, NULL, gws, lws, 0, NULL,
                                     NULL);
        if (ret != 0) {
                fprintf(stderr, "Can't enqueue kernel. Error: %d\n", ret);
                exit(1);
        }

        /* Butterfly Operation */
        fftCore(xmobj, rmobj, wmobj, m, inverse);

        /* Read data from memory buffer */
        ret = clEnqueueReadBuffer(queue, xmobj, CL_TRUE, 0,
                                  n * n * sizeof(cl_float2), xm, 0, NULL, NULL);
        if (ret != 0) {
                fprintf(stderr, "Can't enqueue buffer read. Error: %d\n", ret);
                exit(1);
        }

        float *ampd;
        ampd = (float *)malloc(n * n * sizeof(float));
        for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                        ampd[n * ((i)) + ((j))] =
                            (AMP(((float *)xm)[(2 * n * i) + 2 * j],
                                 ((float *)xm)[(2 * n * i) + 2 * j + 1]));
                }
        }
        opgm.width = n;
        opgm.height = n;
        normalizeF2PGM(&opgm, ampd);
        free(ampd);

        /* Write out image */
        writePGM(&opgm, "output.pgm");

        /* Finalizations*/
        ret = clFlush(queue);
        if (ret != 0) {
                fprintf(stderr, "Can't flush queue. Error: %d\n", ret);
                exit(1);
        }
        ret = clFinish(queue);
        if (ret != 0) {
                fprintf(stderr, "Can't finish queue. Error: %d\n", ret);
                exit(1);
        }
        ret = clReleaseKernel(hpfl);
        if (ret != 0) {
                fprintf(stderr, "Can't release kernel. Error: %d\n", ret);
                exit(1);
        }
        ret = clReleaseKernel(trns);
        if (ret != 0) {
                fprintf(stderr, "Can't release kernel. Error: %d\n", ret);
                exit(1);
        }
        ret = clReleaseKernel(sfac);
        if (ret != 0) {
                fprintf(stderr, "Can't release kernel. Error: %d\n", ret);
                exit(1);
        }
        ret = clReleaseProgram(program);
        if (ret != 0) {
                fprintf(stderr, "Can't release program. Error: %d\n", ret);
                exit(1);
        }
        ret = clReleaseMemObject(xmobj);
        if (ret != 0) {
                fprintf(stderr, "Can't release memory object. Error: %d\n",
                        ret);
                exit(1);
        }
        ret = clReleaseMemObject(rmobj);
        if (ret != 0) {
                fprintf(stderr, "Can't release memory object. Error: %d\n",
                        ret);
                exit(1);
        }
        ret = clReleaseMemObject(wmobj);
        if (ret != 0) {
                fprintf(stderr, "Can't release memory object. Error: %d\n",
                        ret);
                exit(1);
        }
        ret = clReleaseCommandQueue(queue);
        if (ret != 0) {
                fprintf(stderr, "Can't release queue. Error: %d\n", ret);
                exit(1);
        }
        ret = clReleaseContext(context);
        if (ret != 0) {
                fprintf(stderr, "Can't release context. Error: %d\n", ret);
                exit(1);
        }

        destroyPGM(&ipgm);
        destroyPGM(&opgm);

        free(source_str);
        free(wm);
        free(rm);
        free(xm);

        return 0;
}
