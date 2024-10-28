# trl_classifier

## Установка

1. Клонируйте репозиторий:

    ```bash
    git clone https://github.com/PKurnikov/trl_classifier.git
    cd trl_classifier
    ```

2. Установите зависимости:

    ```bash
    pip install -r requirements.txt
    ```

## Использование

1. **Запуск обучения**: Основной скрипт для запуска обучения:

    ```bash
    python train.py
    ```

2. **Экспорт в ONNX**: После завершения обучения можно экспортировать модель:

    ```bash
    python export_onnx.py
    ```

## Настройки

Все настройки обучения и конфигурация модели задаются в `config.py`. Вы можете изменить следующие параметры:

- `data_dir`: Путь к папке с изображениями.
- `batch_size`, `learning_rate`, `num_epochs` и другие гиперпараметры.
- Архитектура модели и параметры аугментаций.

## Данные

Датасет должен быть организован в виде папок, где каждая папка соответствует отдельному классу, например:

data/ ├── car_forward/ ├── car_forward_left/ ├── car_forward_right/ ├── car_stop/ ├── flipped/ ├── ped_forward/ └── ped_stop/