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
    python main.py
    ```

## Настройки

Все настройки обучения и конфигурация модели задаются в `config.py`. Вы можете изменить следующие параметры:

- `data_dir`: Путь к папке с изображениями.
- `batch_size`, `learning_rate`, `num_epochs` и другие гиперпараметры.
- Архитектура модели и параметры аугментаций.

## Данные

Датасет должен быть организован в виде папок, где каждая папка соответствует отдельному классу, например:

data/traffic_lights/
├── car_forward/ 
├── car_forward_left/ 
├── car_forward_right/ 
├── car_stop/ 
├── flipped/ 
├── ped_forward/ 
└── ped_stop/

Датасет доступен по адресу:
https://drive.google.com/file/d/1BYCCLrcA4sQ0MtOV2OF8J2ZeJ_E3fURl/view?usp=drive_link

Необходимо распаковать и добавить в проект в настройках config.py:
data_dir = "./data/traffic_lights"