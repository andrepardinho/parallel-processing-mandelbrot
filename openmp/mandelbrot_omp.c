/*
 * Compilar:
 * gcc -O3 -Wall -Wextra -std=c11 -fopenmp mandelbrot_omp.c -o mandelbrot_omp -lm
 *
 * Exemplos:
 * OMP_NUM_THREADS=1 OMP_SCHEDULE="static,1" \
 * ./mandelbrot_omp padrao 4096
 *
 * OMP_NUM_THREADS=4 OMP_SCHEDULE="dynamic,16" \
 * ./mandelbrot_omp padrao 4096
 *
 * OMP_NUM_THREADS=8 OMP_SCHEDULE="guided,64" \
 * ./mandelbrot_omp cavalos 4096
 *
 * Se OMP_SCHEDULE não for informado, é utilizado static.
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <omp.h>

// Retorna o nome da política de escalonamento.
static const char *nome_schedule(omp_sched_t schedule) {
    switch (schedule) {
        case omp_sched_static:
            return "static";

        case omp_sched_dynamic:
            return "dynamic";

        case omp_sched_guided:
            return "guided";

        case omp_sched_auto:
            return "auto";

        default:
            return "static";
    }
}

/**
 * Calcula o conjunto de Mandelbrot para uma região específica
 * do plano complexo utilizando OpenMP.
 *
 * Os tempos individuais de cada thread são armazenados em
 * tempos_threads para posterior cálculo do fator de balanceamento.
 */
void calcular_mandelbrot(int32_t *count, int width, int height, int max_iter, double re_min, 
    double re_max, double im_min, double im_max, double *tempos_threads, int *num_threads) {

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();

        if (tid == 0) 
            *num_threads = omp_get_num_threads();

        // Mede somente o tempo em que a thread está executando o laço de cálculo.
        double inicio_thread = omp_get_wtime();

        #pragma omp for schedule(runtime) nowait
        for (int py = 0; py < height; py++) {
            for (int px = 0; px < width; px++) {

                // Mapeia o pixel (px, py) para o plano complexo.
                double c_real = re_min + ((double)px / width) * (re_max - re_min);
                double c_imag = im_max - ((double)py / height) * (im_max - im_min);

                double z_real = 0.0;  double z_imag = 0.0;
                int iter = 0;

                while (iter < max_iter) {
                
                double z_real_sq = z_real * z_real;
                double z_imag_sq = z_imag * z_imag;

                if (z_real_sq + z_imag_sq > 4.0) break;
                
                double novo_z_imag = 2.0 * z_real * z_imag + c_imag; // (2ab) + c_imag
                z_real = z_real_sq - z_imag_sq + c_real; // (a^2 - b^2) + c_real
                z_imag = novo_z_imag;
                iter++;
            }
            count[(size_t)py * width + px] = iter;
            }
        }
        
        // nowait permite que cada thread registre seu próprio 
        // tempo assim que terminar seu trabalho.
        tempos_threads[tid] = omp_get_wtime() - inicio_thread;
    }
}

void salvar_binario(const char *arquivo, int32_t *count, int width, int height) {
    FILE *f_bin = fopen(arquivo, "wb");
    if (!f_bin) {
        fprintf(stderr, "Erro ao abrir '%s' para escrita.\n", arquivo);
        return;
    }
    size_t n = (size_t)width * height;
    if (fwrite(count, sizeof(int32_t), n, f_bin) != n) {
        fprintf(stderr, "Erro ao escrever dados binários em '%s'.\n", arquivo);
    }
    fclose(f_bin);
}

static void mapa_de_cor(int iter, int max_iter, unsigned char *r, unsigned char *g, unsigned char *b) {
    if (iter >= max_iter) {
        *r = 11; *g = 29; *b = 58;
        return;
    }
    
    // Visual mais interessante para vista completa e menos interessante para cavalos marinhos:
    //double t = log(1.0 + iter) / log(1.0 + max_iter);
    
    // Visual mais interessante para cavalos marinhos e menos interessante para vista completa:
    double t = (double)(iter % 256) / 256.0;

    double um_menos_t = 1.0 - t;
    double rd = 9.0 * um_menos_t * t * t * t;
    double gd = 15.0 * um_menos_t * um_menos_t * t * t;
    double bd = 8.5 * um_menos_t * um_menos_t * um_menos_t * t;
    *r = (unsigned char)(255.0 * rd);
    *g = (unsigned char)(255.0 * gd);
    *b = (unsigned char)(255.0 * bd);
}

void salvar_imagem_ppm(const char *arquivo, int32_t *count, int width, int height, int max_iter) {
    FILE *f_ppm = fopen(arquivo, "wb");
    if (!f_ppm) {
        fprintf(stderr, "Erro ao abrir '%s' para escrita.\n", arquivo);
        return;
    }
    fprintf(f_ppm, "P6\n%d %d\n255\n", width, height);
    size_t n = (size_t)width * height;
    for (size_t i = 0; i < n; i++) {
        unsigned char r, g, b;
        mapa_de_cor(count[i], max_iter, &r, &g, &b);
        fputc(r, f_ppm);
        fputc(g, f_ppm);
        fputc(b, f_ppm);
    }
    fclose(f_ppm);
}

static double calcular_fator_balanceamento(double *tempos_threads, int num_threads, double *out_t_min, double *out_t_max) {
    if (num_threads <= 0) return 0.0;

    double t_min = tempos_threads[0];
    double t_max = tempos_threads[0];

    for (int i = 1; i < num_threads; i++) {
        if (tempos_threads[i] < t_min) t_min = tempos_threads[i];
        if (tempos_threads[i] > t_max) t_max = tempos_threads[i];
    }

    // Salva os resultados nos endereços de memória passados pelo main
    *out_t_min = t_min;
    *out_t_max = t_max;

    if (t_max <= 0.0) return 0.0;
    return (t_max - t_min) / t_max;
}


int main(int argc, char *argv[]) {

    // Configurações padrão: vista completa
    int width = 4096; int height = 4096; int max_iter = 1000;
    double re_min = -2.0; double re_max = 1.0;
    double im_min = -1.5; double im_max = 1.5;

    const char *prefixo_arquivo = "vista_completa";

    // Leitura do tipo de teste
    if (argc > 1) {
        if (strcmp(argv[1], "cavalos") == 0) {

            max_iter = 5000;

            double cx = -0.743643887; double cy = 0.131825904;
            double half = 3.0e-3 / 2.0;

            re_min = cx - half; re_max = cx + half;
            im_min = cy - half; im_max = cy + half;

            prefixo_arquivo = "cavalos_marinhos";
        }
        else if (strcmp(argv[1], "padrao") != 0) {
            printf("Uso: %s [padrao | cavalos] [resolucao]\n", argv[0]);
            return 1;
        }
    }

    // Leitura da resolução
    if (argc > 2) {
        width = atoi(argv[2]); height = width;

        if (width <= 0) {
            printf("Erro: a resolucao deve ser maior que zero.\n");
            return 1;
        }
    }

    // Garante uma política padrão quando OMP_SCHEDULE não foi definida pelo usuário.
    if (getenv("OMP_SCHEDULE") == NULL)
        omp_set_schedule(omp_sched_static, 0);

    // Obtém informações sobre o escalonamento efetivamente utilizado pelo OpenMP.
    omp_sched_t schedule;
    int chunk;
    omp_get_schedule(&schedule, &chunk);

    // Número máximo de threads que o ambiente poderá utilizar.
    int max_threads = omp_get_max_threads();

    // Vetor para armazenar o tempo de cada thread.
    double *tempos_threads = (double *)calloc((size_t)max_threads, sizeof(double));

    if (tempos_threads == NULL) {
        fprintf( stderr, "Erro: nao foi possivel alocar memoria para os tempos das threads.\n");
        return 1;
    }

    int32_t *count = (int32_t *)malloc((size_t)width *height *sizeof(int32_t));

    if (count == NULL) {
        printf("Erro: nao foi possivel alocar memoria.\n");
        free(tempos_threads);
        return 1;
    }

    // Cria a pasta de saída.
    const char *pasta_saida = "saida";
    mkdir(pasta_saida, 0777);

    // Medição do cálculo
    double t_calc_ini = omp_get_wtime();
    int num_threads = 0;

    calcular_mandelbrot( count, width, height, max_iter, re_min, re_max, im_min, im_max,
        tempos_threads, &num_threads);

    double t_calc_fim = omp_get_wtime();

    // Medição do I/O
    double t_io_ini = omp_get_wtime();

    char caminho_bin[256];
    char caminho_ppm[256];

    snprintf(
        caminho_bin, sizeof(caminho_bin), "%s/mandelbrot_%s_%d.bin",
        pasta_saida, prefixo_arquivo, width);

    snprintf(
        caminho_ppm,sizeof(caminho_ppm), "%s/mandelbrot_%s_%d.ppm",
        pasta_saida, prefixo_arquivo, width);

    salvar_binario(caminho_bin, count, width, height );
    salvar_imagem_ppm(caminho_ppm, count, width, height,max_iter);

    double t_io_fim = omp_get_wtime();

    // Metricas
    double tempo_calc = t_calc_fim - t_calc_ini;
    double tempo_io = t_io_fim - t_io_ini;
    double tempo_total = tempo_calc + tempo_io;
    
    // Variáveis que receberão os tempos mínimo e máximo
    double t_min = 0.0; double t_max = 0.0;
    
    double fator_balanceamento = calcular_fator_balanceamento(tempos_threads, num_threads, &t_min, &t_max);

    // RESULTADOS
    printf("\n");
    printf("========================================\n");
    printf("         BENCHMARK MANDELBROT OMP\n");
    printf("========================================\n");

    printf("Teste:             %s\n", prefixo_arquivo);
    printf("Resolucao:         %dx%d\n", width, height);
    printf("MAX_ITER:          %d\n", max_iter);
    printf("Threads utilizadas: %d\n", num_threads);
    printf("Schedule:          %s\n", nome_schedule(schedule));

    if (chunk > 0) {
        printf("Chunk:             %d\n", chunk);
    } else {
        printf("Chunk:             padrao\n");
    }

    printf("----------------------------------------\n");
    printf("Tempo de calculo:  %.6f s\n", tempo_calc);
    printf("Tempo de I/O:      %.6f s\n", tempo_io);
    printf("Tempo total:       %.6f s\n", tempo_total);
    printf("----------------------------------------\n");
    
    printf("Tempos por thread:\n");
    for (int i = 0; i < num_threads; i++) {
        printf("  Thread %d: %.6f s\n", i, tempos_threads[i]);
    }
    
    printf("----------------------------------------\n");
    printf("T_min:             %.6f s\n", t_min);
    printf("T_max:             %.6f s\n", t_max);
    printf("Fator F_LB:        %.6f\n", fator_balanceamento);
    printf("----------------------------------------\n");
    printf("Binario:           %s\n", caminho_bin);
    printf("Imagem PPM:        %s\n", caminho_ppm);
    printf("========================================\n");

    free(count);
    free(tempos_threads);

    return 0;
}