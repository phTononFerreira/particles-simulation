# ✨ Simulação de Partículas
Este projeto é uma simulação de partículas interativas utilizando Pygame para renderização e Numba para otimização das cálculos de interação. A simulação inclui partículas de diferentes tipos, cada uma com uma cor e comportamento específicos definidos por uma tabela de interações.

![Captura de tela 2024-08-02 100101](https://github.com/user-attachments/assets/e608518a-2d40-43b8-a593-4145e2fda214)

![main-video](https://github.com/user-attachments/assets/6d19dfb2-b507-4308-9ea2-f74322b1d59a)

## ⚙️ Funcionamento
A simulação consiste em partículas que interagem entre si de acordo com uma tabela de interações aleatória. As partículas são desenhadas em uma janela Pygame e as forças de interação são calculadas usando JIT (Just-In-Time) compilation com Numba para acelerar os cálculos na CPU e otimizar a performance.

## 🔧 Métodos Utilizados
- Cálculo de Interações entre Partículas: Utilizamos uma tabela de interações para definir a força e direção das forças entre diferentes tipos de partículas.
- Espaço Toroidal: As partículas se movem em um espaço toroidal, ou seja, quando uma partícula sai de um lado da tela, ela reaparece do lado oposto.
- Tabela de Interações: Uma matriz que define a força de interação entre cada par de tipos de partículas.
- Otimização com JIT: Utilizamos Numba para compilar funções críticas em tempo de execução, aumentando a velocidade dos cálculos.
- Uso do Pygame para Renderização: Pygame é utilizado para desenhar as partículas e a interface gráfica.

![image](https://github.com/user-attachments/assets/e1af95f9-4de9-449f-8c2b-251eba478f65)
