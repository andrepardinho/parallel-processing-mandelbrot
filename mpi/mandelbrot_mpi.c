#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>

#include <mpi.h>

/** 
 * Calcula o conjunto de Mandelbrot para uma região específica do plano complexo.
 * @param count Array para armazenar o número de iterações para cada pixel.
 * @param width Largura da imagem.
 * @param height Altura da imagem.
 * @param max_iter Número máximo de iterações.
 * @param re_min Limite mínimo da parte real do plano complexo.
 * @param re_max Limite máximo da parte real do plano complexo.
 * @param im_min Limite mínimo da parte imaginária do plano complexo.
 * @param im_max Limite máximo da parte imaginária do plano complexo.
 */
void calcular_mandelbrot_mpi_bloco_gather(int32_t *count, int width, int height, int max_iter,
    double re_min, double re_max, double im_min, double im_max, int rank, int np) {

    //calculando as matrizes menores para utilizar o gather depois
    //ESTUDAR GATHERV DEPOIS
    int lim_inf = (rank)*(height/np);
    int lim_sup = (rank+1)*(height/np);
    for (int py = lim_inf; py < lim_sup; py++) {
        for (int px = 0; px < width; px++) {

            double c_real = re_min + ((double)px / width) * (re_max - re_min);
            double c_imag = im_max - ((double)py / height) * (im_max - im_min);

            double z_real = 0.0, z_imag = 0.0;
            int iter = 0;

            while (iter < max_iter) {
                // Condição de escape: |z|^2 > 4.0 (equivalente a |z| > 2,
                // evitando a raiz quadrada).
                double z_real_sq = z_real * z_real;
                double z_imag_sq = z_imag * z_imag;

                if (z_real_sq + z_imag_sq > 4.0) break;
                
                double novo_z_imag = 2.0 * z_real * z_imag + c_imag; // (2ab) + c_imag
                z_real = z_real_sq - z_imag_sq + c_real; // (a^2 - b^2) + c_real
                z_imag = novo_z_imag;
                iter++;
            }
            count[(size_t)(py - lim_inf) * width + px] = iter;
        }
    }
}

void calcular_mandelbrot_mpi_bloco_gatherV(int32_t *count, int width, int height, int max_iter,
    double re_min, double re_max, double im_min, double im_max, int rank, int np) {
    
    // Calculando a quantidade de linhas de cada processo
    int linhas_base = height / np;
    int resto = height % np;

    int linhas_local = linhas_base + (rank < resto ? 1 : 0);

    int lim_inf = rank * linhas_base
                + (rank < resto ? rank : resto);

    int lim_sup = lim_inf + linhas_local;

    for (int py = lim_inf; py < lim_sup; py++) {
        for (int px = 0; px < width; px++) {

            double c_real = re_min
                + ((double)px / width) * (re_max - re_min);

            double c_imag = im_max
                - ((double)py / height) * (im_max - im_min);

            double z_real = 0.0, z_imag = 0.0;
            int iter = 0;

            while (iter < max_iter) {
                double z_real_sq = z_real * z_real;
                double z_imag_sq = z_imag * z_imag;

                if (z_real_sq + z_imag_sq > 4.0)
                    break;

                double novo_z_imag =
                    2.0 * z_real * z_imag + c_imag;

                z_real = z_real_sq - z_imag_sq + c_real;
                z_imag = novo_z_imag;

                iter++;
            }

            // Índice local: a primeira linha calculada é a linha 0 do buffer
            count[(size_t)(py - lim_inf) * width + px] = iter;
        }
    }
}

void calcular_mandelbrot_mpi_intercalado_gatherV(int32_t *count, int width, int height, int max_iter,
    double re_min, double re_max, double im_min, double im_max, int rank, int np, int chunk) {

    int linha_local = 0;

    // Percorre os blocos de linhas da imagem
    for (int bloco = 0; bloco * chunk < height; bloco++) {

        // Processo responsável por este bloco
        int processo_bloco = bloco % np;

        if (processo_bloco != rank)
            continue;

        int lim_inf = bloco * chunk;
        int lim_sup = lim_inf + chunk;

        if (lim_sup > height)
            lim_sup = height;

        // Calcula as linhas globais deste bloco
        for (int py = lim_inf; py < lim_sup; py++) {
            for (int px = 0; px < width; px++) {

                double c_real = re_min
                    + ((double)px / width) * (re_max - re_min);

                double c_imag = im_max
                    - ((double)py / height) * (im_max - im_min);

                double z_real = 0.0, z_imag = 0.0;
                int iter = 0;

                while (iter < max_iter) {
                    double z_real_sq = z_real * z_real;
                    double z_imag_sq = z_imag * z_imag;

                    if (z_real_sq + z_imag_sq > 4.0)
                        break;

                    double novo_z_imag =
                        2.0 * z_real * z_imag + c_imag;

                    z_real = z_real_sq - z_imag_sq + c_real;
                    z_imag = novo_z_imag;

                    iter++;
                }

                // Armazena na próxima linha disponível do buffer local
                count[(size_t)linha_local * width + px] = iter;
            }

            linha_local++;
        }
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

    // Normalização cores fundo
    
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

static double tempo_atual_segundos(void) {
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}




int main(int argc, char *argv[]) {

    //definindo entradas padrao
    const char *schedule = "bloco";    
    int chunk = 1;

    // Configurações padrão: vista completa
    int width = 4096; int height = 4096; int max_iter = 1000;
    double re_min = -2.0; double re_max = 1.0;
    double im_min = -1.5; double im_max = 1.5;

    //Configuracoes mpi
    int rank, np;
    //int orig;
    //MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);


    const char *prefixo_arquivo = "vista_completa";

    //Verificando quantidade de argumentos
    if (argc > 5) {
        printf("Erro: muitos argumentos\n")
        printf("Uso: %s [padrao | cavalos] [resolucao] [schedule] [chunk]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

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

            printf("Uso: %s [padrao | cavalos] [resolucao] [schedule] [chunk]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Leitura da resolução
    if (argc > 2) {
        width = atoi(argv[2]); height = width;

        if (width <= 0) {
            printf("Erro: a resolucao deve ser maior que zero.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Leitura do schedule
    if (argc > 3) {
        if (strcmp(argv[3], "bloco") != 0 && 
            strcmp(argv[3], "blocov") != 0 && 
            strcmp(argv[3], "intercaladov") != 0
        ) {
            printf("Erro: Schedule deve ser = bloco | blocov | intercaladov");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        else
            schedule = argv[3];
    }

    if (strcmp(schedule, "bloco") == 0 && height % np != 0) {
    printf("Erro: para o schedule 'bloco', a altura deve ser divisivel pelo numero de processos.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
}

    //leitura de chunk
    if (argc > 4) {
        if(strcmp(schedule, "intercaladov") != 0){
            printf("Erro: entrada de chunk permitida apenas com schedule 'intercaladov'\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if(atoi(argv[4]) <= 0 || atoi(argv[4]) > height){
            printf("Erro: valor de chunk deve ser maior que 0 e menor ou igual a resolucao\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        chunk = atoi(argv[4]);
    }

    int linhas_base = height / np;
    int resto = height % np;

    int linhas_local = 0;

    if (strcmp(schedule, "bloco") == 0)
        linhas_local = height / np;
    else if (strcmp(schedule, "blocov") == 0)
        linhas_local = linhas_base + (rank < resto ? 1 : 0);
    else if (strcmp(schedule, "intercaladov") == 0) {
        int total_blocos = (height + chunk - 1) / chunk;

        for (int bloco = rank; bloco < total_blocos; bloco += np) {
            int inicio = bloco * chunk;
            int fim = inicio + chunk;

            if (fim > height)
                fim = height;

            linhas_local += fim - inicio;
        }
    }

    size_t linhas_alocadas = (size_t)(linhas_local > 0 ? linhas_local : 1);
    int32_t *count_local = malloc((size_t)width * linhas_alocadas * sizeof(int32_t));
    if (count_local == NULL) {
        fprintf(stderr, "Erro: nao foi possivel alocar memoria.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int32_t *count_global = NULL;
    
    if(rank == 0){
        count_global =
            (int32_t *)malloc((size_t)width * height * sizeof(int32_t));

        if (count_global == NULL) {
            printf("Erro: nao foi possivel alocar memoria.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        recvcounts = malloc((size_t)np * sizeof(int));
        displs = malloc((size_t)np * sizeof(int));

        if (recvcounts == NULL || displs == NULL) {
            fprintf(stderr, "Erro ao alocar recvcounts/displs.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int deslocamento = 0;

        for (int i = 0; i < np; i++) {
            int linhas_i = 0;

            if (strcmp(schedule, "intercaladov") == 0) {
                int total_blocos = (height + chunk - 1) / chunk;

                for (int bloco = i; bloco < total_blocos; bloco += np) {
                    int inicio = bloco * chunk;
                    int fim = inicio + chunk;

                    if (fim > height)
                        fim = height;

                    linhas_i += fim - inicio;
                }

                // Recebimento em um buffer compacto, por processo.
                recvcounts[i] = linhas_i * width;
                displs[i] = deslocamento;
                deslocamento += recvcounts[i];
            }
            else {
                linhas_i = height / np + (i < resto ? 1 : 0);

                int inicio_i = i * linhas_base
                            + (i < resto ? i : resto);

                recvcounts[i] = linhas_i * width;
                displs[i] = inicio_i * width;
            }
        }
    }

    // Medição do cálculo
    double t_calc_ini_global;
    double t_calc_fim_global;

    double t_calc_ini_local = tempo_atual_segundos();

    if(strcmp(schedule, "bloco") == 0)
        calcular_mandelbrot_mpi_bloco_gather( 
            count_local, width, height, max_iter, 
            re_min, re_max, im_min,im_max,
            rank, np
        );
    else if (strcmp(schedule, "blocov") == 0)
        calcular_mandelbrot_mpi_bloco_gatherV( 
            count_local, width, height, max_iter, 
            re_min, re_max, im_min,im_max,
            rank, np
        );
    else if (strcmp(schedule, "intercaladov") == 0)
        calcular_mandelbrot_mpi_intercalado_gatherV( 
            count_local, width, height, max_iter, 
            re_min, re_max, im_min,im_max,
            rank, np, chunk
        );

    double t_calc_fim_local = tempo_atual_segundos();

    MPI_Reduce(&t_calc_ini_local, &t_calc_ini_global, 
        1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    MPI_Reduce(&t_calc_fim_local, &t_calc_fim_global,
        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int32_t *count_intercalado = NULL;

    if (rank == 0 &&
        strcmp(schedule, "intercaladov") == 0) {

        count_intercalado = malloc(
            (size_t)width * height * sizeof(int32_t)
        );

        if (count_intercalado == NULL) {
            fprintf(stderr, "Erro ao alocar buffer intercalado.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (strcmp(schedule, "bloco") == 0) {
        MPI_Gather(
            count_local, width * linhas_local, MPI_INT32_T,
            count_global, width * linhas_local, MPI_INT32_T,
            0, MPI_COMM_WORLD
        );
    }
    else if (strcmp(schedule, "blocov") == 0) {
        MPI_Gatherv(
            count_local, width * linhas_local, MPI_INT32_T,
            count_global, recvcounts, displs, MPI_INT32_T,
            0, MPI_COMM_WORLD
        );
    }
    else if (strcmp(schedule, "intercaladov") == 0) {
        MPI_Gatherv(
            count_local, width * linhas_local, MPI_INT32_T,
            count_intercalado, recvcounts, displs, MPI_INT32_T,
            0, MPI_COMM_WORLD
        );

        if (rank == 0) {
            int total_blocos = (height + chunk - 1) / chunk;

            // Reorganiza os segmentos compactos na ordem global das linhas.
            for (int processo = 0; processo < np; processo++) {
                int linha_local = 0;

                for (int bloco = processo;
                    bloco < total_blocos;
                    bloco += np) {

                    int inicio = bloco * chunk;
                    int fim = inicio + chunk;

                    if (fim > height)
                        fim = height;

                    for (int py = inicio; py < fim; py++) {
                        memcpy(
                            &count_global[(size_t)py * width],
                            &count_intercalado[
                                (size_t)displs[processo]
                                + (size_t)linha_local * width
                            ],
                            (size_t)width * sizeof(int32_t)
                        );

                        linha_local++;
                    }
                }
            }
        }
    }

    // Salvamento
    if(rank == 0){
        // Cria a pasta de saída
        const char *pasta_saida = "saida";

        //para windows:
        //mkdir(pasta_saida);

        //para linux
        mkdir(pasta_saida, 0777);
    
        double t_io_ini = tempo_atual_segundos();
        char caminho_bin[256];
        char caminho_ppm[256];

        snprintf(
            caminho_bin, sizeof(caminho_bin),
            "%s/mandelbrot_%s_%d.bin", 
            pasta_saida, prefixo_arquivo, width );

        snprintf(
            caminho_ppm, sizeof(caminho_ppm),
            "%s/mandelbrot_%s_%d.ppm",
            pasta_saida, prefixo_arquivo, width );

        salvar_binario(caminho_bin, count_global, width, height);
        salvar_imagem_ppm(caminho_ppm, count_global, width, height, max_iter);

        double t_io_fim = tempo_atual_segundos();

        // Resultados
        printf("\n== Teste: %s ==\n", prefixo_arquivo);
        printf("Resolucao: %dx%d | MAX_ITER: %d\n", width, height, max_iter);
        printf("Tempo de calculo: %.6f s\n", t_calc_fim_global - t_calc_ini_global);
        printf("Tempo de I/O:     %.6f s\n", t_io_fim - t_io_ini);
        printf("Tempo total:      %.6f s\n", 
            (t_calc_fim_global - t_calc_ini_global) + (t_io_fim - t_io_ini));

    }
    
    free(count_local);
    if (rank == 0) {
        free(count_global);
        free(recvcounts);
        free(displs);
        free(count_intercalado);
    }

    MPI_Finalize();
    return 0;
}