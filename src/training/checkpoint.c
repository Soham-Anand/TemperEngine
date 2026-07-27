#include "temper/training/checkpoint.h"
#include "temper/core/logger.h"
#include <stdio.h>
#include <string.h>

int temper_checkpoint_save(const char *path, const TemperCheckpoint *cp)
{
    FILE *f = fopen(path, "wb");
    if (!f)
    {
        temper_error("Failed to save checkpoint: %s", path);
        return -1;
    }
    fwrite(cp, sizeof(TemperCheckpoint), 1, f);
    fclose(f);
    temper_info("Checkpoint saved: %s", path);
    return 0;
}

int temper_checkpoint_load(const char *path, TemperCheckpoint *cp)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        temper_error("Failed to load checkpoint: %s", path);
        return -1;
    }
    fread(cp, sizeof(TemperCheckpoint), 1, f);
    fclose(f);
    temper_info("Checkpoint loaded: %s", path);
    return 0;
}

bool temper_checkpoint_exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f)
    {
        fclose(f);
        return true;
    }
    return false;
}
