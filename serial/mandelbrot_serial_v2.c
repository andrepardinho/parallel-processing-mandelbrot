#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
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
 */
void calcular_mandelbrot(int *count, int width, int height, int max_iter,
    double re_min, double re_max, double im_min, double im_max) {

    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {

            // Mapeia o pixel (px, py) para o plano complexo.
            // Exemplo: se px = 2048 e width = 4096, (double)px / width = 0.5,
            // ou seja, o meio do intervalo horizontal. Com re_min = -2.0 e
            // re_max = 1.0: c_real = -2.0 + 0.5 * (1.0 - (-2.0)) = -0.5.
            // mesma logica para c_imag usando im_min/im_max e py/height.
            double c_real = re_min + ((double)px / width) * (re_max - re_min);
            //double c_imag = im_min + ((double)py / height) * (im_max - im_min);
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
            count[(size_t)py * width + px] = iter;
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
    double t = log(1.0 + iter) / log(1.0 + max_iter);
    //double t = (double)(iter % 256) / 256.0;

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

    // Cria a pasta de saída
    const char *pasta_saida = "saida";
    mkdir(pasta_saida, 0777);

    int32_t *count =
        (int32_t *)malloc((size_t)width * height * sizeof(int32_t));

    if (count == NULL) {
        printf("Erro: nao foi possivel alocar memoria.\n");
        return 1;
    }

    // Medição do cálculo
    double t_calc_ini = tempo_atual_segundos();

    calcular_mandelbrot( count, width, height, max_iter, re_min, re_max, im_min,im_max );

    double t_calc_fim = tempo_atual_segundos();

    // Salvamento
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

    salvar_binario(caminho_bin, count, width, height);
    salvar_imagem_ppm(caminho_ppm, count, width, height, max_iter);

    double t_io_fim = tempo_atual_segundos();

    // Resultados
    printf("\n== Teste: %s ==\n", prefixo_arquivo);
    printf("Resolucao: %dx%d | MAX_ITER: %d\n", width, height, max_iter);
    printf("Tempo de calculo: %.6f s\n", t_calc_fim - t_calc_ini);
    printf("Tempo de I/O:     %.6f s\n", t_io_fim - t_io_ini);
    printf("Tempo total:      %.6f s\n", (t_calc_fim - t_calc_ini) + (t_io_fim - t_io_ini));

    free(count);

    return 0;
}