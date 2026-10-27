#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

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
 * @param rank Rank do processo MPI.
 * @param np Número de processos MPI.
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

/**
 * Versão com Gatherv: distribui as linhas entre os processos considerando
 * o resto da divisão, de modo que todos os processos recebam uma carga
 * balanceada mesmo quando height não é divisível por np.
 */
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
                // Condição de escape: |z|^2 > 4.0 (equivalente a |z| > 2,
                // evitando a raiz quadrada).
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

/**
 * Versão intercalada com Gatherv: os blocos de linhas são distribuídos
 * ciclicamente entre os processos (bloco % np), e cada processo armazena
 * suas linhas de forma compacta no buffer local.
 */
void calcular_mandelbrot_mpi_intercalado_gatherV(int32_t *count, int width, int height, int max_iter,
    double re_min, double re_max, double im_min, double im_max, int rank, int np, int chunk) {

    int linha_local = 0;
    int total_blocos = 1 + (height - 1) / chunk;

    // Cada processo recebe blocos de chunk linhas em ordem cíclica.
    for (int bloco = rank; bloco < total_blocos; bloco += np) {

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
                    // Condição de escape: |z|^2 > 4.0 (equivalente a |z| > 2,
                    // evitando a raiz quadrada).
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

/**
 * Aborta a execução MPI imprimindo a mensagem apenas no rank 0.
 */
static void abortar_com_erro(int rank, const char *mensagem) {
    if (rank == 0) {
        fprintf(stderr, "%s\n", mensagem);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

/**
 * Lê um inteiro positivo da string, retornando 1 em caso de sucesso.
 */
static int ler_inteiro_positivo(const char *texto, int *valor) {
    char *fim = NULL;
    errno = 0;
    long convertido = strtol(texto, &fim, 10);

    if (errno == ERANGE || fim == texto || *fim != '\0' ||
        convertido <= 0 || convertido > INT_MAX) {
        return 0;
    }

    *valor = (int)convertido;
    return 1;
}

/**
 * Salva o buffer de iterações em arquivo binário.
 */
static int salvar_binario(const char *arquivo, const int32_t *count,
                          int width, int height) {
    FILE *f_bin = fopen(arquivo, "wb");
    if (f_bin == NULL) {
        fprintf(stderr, "Erro ao abrir '%s' para escrita: %s\n", arquivo, strerror(errno));
        return 0;
    }

    size_t n = (size_t)width * (size_t)height;
    int ok = 1;
    if (fwrite(count, sizeof(int32_t), n, f_bin) != n) {
        fprintf(stderr, "Erro ao escrever dados binarios em '%s'.\n", arquivo);
        ok = 0;
    }
    if (fclose(f_bin) != 0) {
        fprintf(stderr, "Erro ao fechar '%s'.\n", arquivo);
        ok = 0;
    }
    return ok;
}

/**
 * Mapeia a iteração para uma cor RGB.
 */
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

/**
 * Salva a imagem em formato PPM (P6).
 */
static int salvar_imagem_ppm(const char *arquivo, const int32_t *count,
                             int width, int height, int max_iter) {
    FILE *f_ppm = fopen(arquivo, "wb");
    if (f_ppm == NULL) {
        fprintf(stderr, "Erro ao abrir '%s' para escrita: %s\n", arquivo, strerror(errno));
        return 0;
    }

    int ok = fprintf(f_ppm, "P6\n%d %d\n255\n", width, height) >= 0;
    size_t n = (size_t)width * (size_t)height;

    for (size_t i = 0; ok && i < n; i++) {
        unsigned char r, g, b;
        mapa_de_cor(count[i], max_iter, &r, &g, &b);

        if (fputc(r, f_ppm) == EOF ||
            fputc(g, f_ppm) == EOF ||
            fputc(b, f_ppm) == EOF) {
            ok = 0;
        }
    }

    if (fclose(f_ppm) != 0) {
        ok = 0;
    }
    if (!ok) {
        fprintf(stderr, "Erro ao escrever ou fechar a imagem '%s'.\n", arquivo);
    }
    return ok;
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
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);


    const char *prefixo_arquivo = "vista_completa";

    //Verificando quantidade de argumentos
    if (argc > 5) {
        if (rank == 0) {
            fprintf(stderr,
                    "Uso: %s [padrao|cavalos] [resolucao] [bloco|blocov|intercaladov] [chunk]\n",
                    argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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

            abortar_com_erro(rank, "Caso invalido. Use 'padrao' ou 'cavalos'.");
        }
    }

    // Leitura da resolução
    if (argc > 2) {
        if (!ler_inteiro_positivo(argv[2], &width)) {
            abortar_com_erro(rank, "Resolucao invalida: informe um inteiro positivo.");
        }
        height = width;
    }

    // Leitura do schedule
    if (argc > 3) {
        if (strcmp(argv[3], "bloco") != 0 &&
            strcmp(argv[3], "blocov") != 0 &&
            strcmp(argv[3], "intercaladov") != 0
        ) {
            abortar_com_erro(rank, "Estrategia invalida. Use 'bloco', 'blocov' ou 'intercaladov'.");
        }
        else
            schedule = argv[3];
    }

    if (strcmp(schedule, "bloco") == 0 && height % np != 0) {
        abortar_com_erro(rank, "A estrategia 'bloco' exige que a resolucao seja divisivel pelo numero de processos.");
    }

    //leitura de chunk
    if (argc > 4) {
        if(strcmp(schedule, "intercaladov") != 0){
            abortar_com_erro(rank, "O argumento chunk so pode ser informado com 'intercaladov'.");
        }

        if(!ler_inteiro_positivo(argv[4], &chunk) || chunk > height){
            abortar_com_erro(rank, "Chunk invalido: informe um inteiro entre 1 e a resolucao.");
        }
    }

    /* MPI_Gatherv usa counts/displacements do tipo int nesta implementacao. */
    if ((size_t)width > SIZE_MAX / (size_t)height) {
        abortar_com_erro(rank, "A resolucao excede os limites de tamanho suportados.");
    }
    size_t total_pixels = (size_t)width * (size_t)height;
    if (total_pixels > (size_t)INT_MAX ||
        total_pixels > SIZE_MAX / sizeof(int32_t)) {
        abortar_com_erro(rank,
            "Matriz grande demais: os counts/deslocamentos do MPI ou a alocacao excederiam os limites suportados.");
    }

    int linhas_base = height / np;
    int resto = height % np;

    int linhas_local = 0;

    if (strcmp(schedule, "bloco") == 0)
        linhas_local = height / np;
    else if (strcmp(schedule, "blocov") == 0)
        linhas_local = linhas_base + (rank < resto ? 1 : 0);
    else if (strcmp(schedule, "intercaladov") == 0) {
        int total_blocos = 1 + (height - 1) / chunk;

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
        fprintf(stderr, "Erro: nao foi possivel alocar buffer local no rank %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int32_t *count_global = NULL;
    int32_t *count_intercalado = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        count_global = malloc(total_pixels * sizeof(int32_t));
        if (count_global == NULL) {
            fprintf(stderr, "Erro: rank 0 nao conseguiu alocar a matriz global.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if (strcmp(schedule, "intercaladov") == 0) {
            // Alocar antes da regiao cronometrada, nao durante a coleta.
            count_intercalado = malloc(total_pixels * sizeof(int32_t));
            if (count_intercalado == NULL) {
                fprintf(stderr, "Erro: rank 0 nao conseguiu alocar o buffer intercalado.\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }

        recvcounts = malloc((size_t)np * sizeof(int));
        displs = malloc((size_t)np * sizeof(int));
        if (recvcounts == NULL || displs == NULL) {
            fprintf(stderr, "Erro: rank 0 nao conseguiu alocar recvcounts/displs.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        int deslocamento = 0;

        for (int i = 0; i < np; i++) {
            int linhas_i = 0;

            if (strcmp(schedule, "intercaladov") == 0) {
                int total_blocos = 1 + (height - 1) / chunk;
                for (int bloco = i; bloco < total_blocos; bloco += np) {
                    int inicio = bloco * chunk;
                    int fim = inicio + chunk;
                    if (fim > height) {
                        fim = height;
                    }
                    linhas_i += fim - inicio;
                }

                // Displacements compactos no buffer recebido por rank.
                recvcounts[i] = linhas_i * width;
                displs[i] = deslocamento;
                deslocamento += recvcounts[i];
            } else {
                linhas_i = linhas_base + (i < resto ? 1 : 0);
                int inicio_i = i * linhas_base + (i < resto ? i : resto);
                recvcounts[i] = linhas_i * width;
                displs[i] = inicio_i * width;
            }
        }
    }

    double tempo_calculo_local_s = 0.0;
    double tempo_calculo_min_s = 0.0;
    double tempo_calculo_max_s = 0.0;
    double tempo_coleta_s = 0.0;
    double tempo_reorganizacao_s = 0.0;
    double tempo_paralelo_total_s = 0.0;

    // O tempo total comeca antes da barreira para incluir a sincronizacao inicial.
    double tempo_total_inicio = 0.0;
    if (rank == 0) {
        tempo_total_inicio = MPI_Wtime();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Medição do cálculo
    double tempo_calculo_inicio = MPI_Wtime();
    if (strcmp(schedule, "bloco") == 0)
        calcular_mandelbrot_mpi_bloco_gather(
            count_local, width, height, max_iter,
            re_min, re_max, im_min, im_max, rank, np);
    else if (strcmp(schedule, "blocov") == 0)
        calcular_mandelbrot_mpi_bloco_gatherV(
            count_local, width, height, max_iter,
            re_min, re_max, im_min, im_max, rank, np);
    else if (strcmp(schedule, "intercaladov") == 0)
        calcular_mandelbrot_mpi_intercalado_gatherV(
            count_local, width, height, max_iter,
            re_min, re_max, im_min, im_max, rank, np, chunk);
    tempo_calculo_local_s = MPI_Wtime() - tempo_calculo_inicio;

    double tempo_coleta_inicio = 0.0;
    if (rank == 0) {
        tempo_coleta_inicio = MPI_Wtime();
    }

    if (strcmp(schedule, "bloco") == 0) {
        MPI_Gather(
            count_local, width * linhas_local, MPI_INT32_T,
            count_global, width * linhas_local, MPI_INT32_T,
            0, MPI_COMM_WORLD);
    }
    else if (strcmp(schedule, "blocov") == 0) {
        MPI_Gatherv(
            count_local, width * linhas_local, MPI_INT32_T,
            count_global, recvcounts, displs, MPI_INT32_T,
            0, MPI_COMM_WORLD);
    }
    else if (strcmp(schedule, "intercaladov") == 0) {
        MPI_Gatherv(
            count_local, width * linhas_local, MPI_INT32_T,
            count_intercalado, recvcounts, displs, MPI_INT32_T,
            0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        tempo_coleta_s = MPI_Wtime() - tempo_coleta_inicio;
    }

    if (rank == 0 && strcmp(schedule, "intercaladov") == 0) {
        double tempo_reorganizacao_inicio = MPI_Wtime();
        int total_blocos = 1 + (height - 1) / chunk;

        // Remonta a matriz global a partir dos buffers compactos por processo.
        for (int processo = 0; processo < np; processo++) {
            int linha_local = 0;
            for (int bloco = processo; bloco < total_blocos; bloco += np) {
                int inicio = bloco * chunk;
                int fim = inicio + chunk;
                if (fim > height) {
                    fim = height;
                }

                for (int py = inicio; py < fim; py++) {
                    memcpy(
                        &count_global[(size_t)py * (size_t)width],
                        &count_intercalado[(size_t)displs[processo]
                            + (size_t)linha_local * (size_t)width],
                        (size_t)width * sizeof(int32_t));
                    linha_local++;
                }
            }
        }
        tempo_reorganizacao_s = MPI_Wtime() - tempo_reorganizacao_inicio;
    }

    // Tempo de parede do rank 0 ate a matriz estar pronta; sem escrita de arquivos.
    if (rank == 0) {
        tempo_paralelo_total_s = MPI_Wtime() - tempo_total_inicio;
    }

    // Reduzir duracoes locais (nao timestamps absolutos entre processos).
    MPI_Reduce(&tempo_calculo_local_s, &tempo_calculo_min_s,
               1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tempo_calculo_local_s, &tempo_calculo_max_s,
               1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int status_saida = 0;
    double tempo_io_s = 0.0;

    // Salvamento
    if (rank == 0) {
        // Cria a pasta de saída
        const char *pasta_saida = "saida";
        struct stat info_pasta;
        int pasta_ok = 1;

        //para windows:
        //mkdir(pasta_saida);

        //para linux
        if (mkdir(pasta_saida, 0777) != 0 && errno != EEXIST) {
            fprintf(stderr, "Erro ao criar a pasta '%s': %s\n",
                    pasta_saida, strerror(errno));
            pasta_ok = 0;
        }
        if (pasta_ok &&
            (stat(pasta_saida, &info_pasta) != 0 || !S_ISDIR(info_pasta.st_mode))) {
            fprintf(stderr, "Erro: '%s' nao e um diretorio valido.\n", pasta_saida);
            pasta_ok = 0;
        }

        if (pasta_ok) {
            char caminho_bin[256];
            char caminho_ppm[256];
            snprintf(caminho_bin, sizeof(caminho_bin),
                     "%s/mandelbrot_%s_%d.bin", pasta_saida, prefixo_arquivo, width);
            snprintf(caminho_ppm, sizeof(caminho_ppm),
                     "%s/mandelbrot_%s_%d.ppm", pasta_saida, prefixo_arquivo, width);

            double tempo_io_inicio = MPI_Wtime();
            int bin_ok = salvar_binario(caminho_bin, count_global, width, height);
            int ppm_ok = salvar_imagem_ppm(caminho_ppm, count_global,
                                           width, height, max_iter);
            tempo_io_s = MPI_Wtime() - tempo_io_inicio;
            if (!bin_ok || !ppm_ok) {
                status_saida = 1;
            }
        } else {
            status_saida = 1;
        }

        double f_lb = 0.0;
        if (tempo_calculo_max_s > 0.0) {
            f_lb = (tempo_calculo_max_s - tempo_calculo_min_s)
                 / tempo_calculo_max_s;
        }

        // Resultados
        printf("\n== Teste: %s ==\n", prefixo_arquivo);
        printf("Resolucao: %dx%d | MAX_ITER: %d\n", width, height, max_iter);
        printf("ESTRATEGIA = %s\n", schedule);
        printf("PROCESSOS = %d\n", np);
        printf("CHUNK = %d\n", strcmp(schedule, "intercaladov") == 0 ? chunk : 0);
        printf("TEMPO_CALCULO_MIN_S = %.9f\n", tempo_calculo_min_s);
        printf("TEMPO_CALCULO_MAX_S = %.9f\n", tempo_calculo_max_s);
        printf("F_LB = %.9f\n", f_lb);
        printf("TEMPO_COLETA_S = %.9f\n", tempo_coleta_s);
        printf("TEMPO_REORGANIZACAO_S = %.9f\n", tempo_reorganizacao_s);
        printf("TEMPO_PARALELO_TOTAL_S = %.9f\n", tempo_paralelo_total_s);
        printf("TEMPO_IO_S = %.9f\n", tempo_io_s);
        printf("TEMPO_TOTAL_COM_IO_S = %.9f\n", tempo_paralelo_total_s + tempo_io_s);
        printf("STATUS_SAIDA = %s\n", status_saida == 0 ? "OK" : "ERRO");
    }

    // Faz todos os ranks retornarem falha se os arquivos nao foram gravados.
    MPI_Bcast(&status_saida, 1, MPI_INT, 0, MPI_COMM_WORLD);

    free(count_local);
    if (rank == 0) {
        free(count_global);
        free(count_intercalado);
        free(recvcounts);
        free(displs);
    }

    MPI_Finalize();
    return status_saida == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}