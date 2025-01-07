# Treinamento IA Base Henet

## Descrição Geral
Este projeto realiza extração, normalização e processamento de dados textuais do site [Henet](https://www.henet.com.br), com foco em consultas semânticas. Utilizando inteligência artificial, o sistema responde perguntas com base nos dados coletados.

## Funcionalidades
- **Web Scraping**: Extração de textos do site Henet.
- **Normalização de Dados**: Processamento e limpeza dos textos coletados.
- **Segmentação e Tokenização**: Divisão dos textos em tokens para análise eficiente.
- **Geração de Embeddings**: Utilização da API ChatGPT para transformar textos em embeddings vetoriais.
- **Armazenamento em Base Vetorial**: Dados processados são armazenados para consultas rápidas e precisas.
- **Consultas Semânticas**: Permite realizar perguntas, com respostas geradas pela IA baseadas nos dados extraídos.

## Informações Técnicas
1. **Extração de Dados**:
   - Textos são coletados do site `henet.com.br` via técnicas de web scraping.
2. **Armazenamento em arquivos**
   - Os dados armazenados pelo site e suas depedências são armazenados em arquivos .txt para organização
3. **Normalização e Transformação**:
   - É feito a leitura dos arquivos .txt e logo após os textos são normalizados para remover inconsistências, formatados e divididos em tokens.
4. **Geração de Embeddings**:
   - A API do ChatGPT transforma os textos em representações vetoriais (embeddings), otimizadas para consultas semânticas.
5. **Armazenamento**:
   - Os embeddings gerados são armazenados em uma base vetorial.
6. **Consultas Semânticas**:
   - Usuários podem consultar o sistema, recebendo respostas baseadas nos dados processados.

## Observação
- O token da API não está presente no repositório pois é uma API custeada. Solicitar ao desenvolvedor.
