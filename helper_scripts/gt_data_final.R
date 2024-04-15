library(data.table)
library(tidyverse)

# adjust every read and write path in this file
gt <- fread("/Users/mathias/Code/PhentypingLoc/everything_data/data/gt.csv") %>%
  mutate(across(everything(), ~str_replace(., "ZEAKJ", "ZEAMX"))) %>%
  mutate(across(everything(), ~str_replace(., "ZEALP", "ZEAMX")))


# tmp <- gt[gt$filename %like% "ARTVU", ]
# tmp_undesired <- tmp[!(label_id %in% "ZEAMX"), ]
# tmp <- tmp[!(tray_id %in% unique(tmp_undesired$tray)), ]
# tmp1 <- gt[gt$filename %like% "ZEAMX", ]
  


ARTVU_train_trays <- c(132804,  132807,  132813,  132816,  132819,  132825,
                      132802,  132805,  132808,  132817,  132829,
                      132803,  132806,  132809,  132812,  132815)
ARTVU_train_incomplete <- c(132810, 132814, 132818, 132820, 132821, 132822, 132823, 132824, 132826, 132827, 132828, 132830, 132832, 132833, 132834)
ARTVU_val_trays <- c(132801)
ARTVU_test_incomplete <- c(132831) # rausgenommen da incomplete
ARTVU_test_trays <- c(132811)


PULDY_train_trays <- c(114811, 114815, 114817, 114810, 114812, 114901)
PULDY_train_incomplete <- c(114801, 114802, 114803, 114804, 114805, 114806, 114807, 114808, 114809, 114813, 114814, 114816)
PULDY_val_trays <-c(114905)
PULDY_test_trays <- c(114904)
# 114902


ZEAMX_train_trays <- c(127801, 127802, 127803, 127804, 127805, 127806, 127807, 127808, 127809, 127810, 127811, 127812,
                       127813, 127814, 127815, 127816, 127817, 127818, 127819, 127820, 127821, 127822, 127823, 127824,
                       127825, 127826, 127827, 127828, 127829, 127830, 127831, 127832, 127833, 127834, 127835, 127836,
                       127837, 139801, 139802, 139803, 139804, 139805, 139806, 139807, 139808, 139809, 139810,
                       139811, 139812, 139813, 139814, 139815, 139816, 139817, 139818, 139819, 139820, 139821, 139822,
                       139823, 139824, 139825, 139826, 139827, 139828, 139829, 139830, 139831, 139832, 139833, 139834,
                       139835, 139836, 139838)
ZEAMX_val_trays <- c(139837)
ZEAMX_test_trays <- c(127838)


desired_labels <- c("ARTVU", "PULDY", "ZEAMX")


check_overlapping <- function(data) {
  overlapping_count <- 0
  n_boxes <- nrow(data)

  if (n_boxes > 1) {
    for (i in 1:(n_boxes - 1)) {
      for (j in (i + 1):n_boxes) {
        # Check if bounding boxes (xmin, ymin, xmax, ymax) overlap
        box1 <- data[i, c("xmin", "ymin", "xmax", "ymax")]
        box2 <- data[j, c("xmin", "ymin", "xmax", "ymax")]

        if (box1["xmax"] >= box2["xmin"] && box2["xmax"] >= box1["xmin"] &&
            box1["ymax"] >= box2["ymin"] && box2["ymax"] >= box1["ymin"]) {
          overlapping_count <- overlapping_count + 1
        }
      }
    }
  }

  data.frame(overlapping_count)
}


# whole train set
train_set <- gt[tray_id %in% c(ARTVU_train_trays, PULDY_train_trays, ZEAMX_train_trays), ]

#fill1 <- unique(train_set$label_id)
#print(fill1)


# whole undesired bounding boxes
undesired_entries <- train_set[!(label_id %in% desired_labels), ]

# tray_id of undesired bounding boxes
undesired_trays <- unique(undesired_entries$tray_id)
print(undesired_trays)


# total # of images per tray_id with undesired images
total_c_undesired_trays <- train_set[tray_id %in% undesired_trays, .(tray_id = tray_id), by=filename] %>%
  distinct(.keep_all = TRUE) %>%
  .[, .N, by=tray_id]

# # of undesired images per tray_id
undesired_entries_count <- undesired_entries[, .(tray_id = tray_id, count = .N), by=filename] %>%
  distinct(.keep_all = TRUE) %>%
  .[, .N, by=tray_id]
undesired_entries_count$total <- total_c_undesired_trays$N

# total # of undesired images
print(sum(undesired_entries_count$N))
print(undesired_entries_count$tray_id)


# train_set without files with # of classes >= 4
cleaned_train_set <- train_set[!(filename %in% unique(undesired_entries$filename))]
# proof
print(unique(cleaned_train_set$label_id))
print(length(unique(cleaned_train_set$filename)))



# DEBUG
debug1 <- length(unique(cleaned_train_set$filename))
print(unique(undesired_entries$filename))
#cat(unique(undesired_entries$filename), file = "/Users/mathias/Code/PhentypingLoc/outfiles/undesired_entries_filename.txt")

# DEBUG


train_stats <- sapply(str_split(unique(cleaned_train_set$filename), "/"), `[`, 1)
print(table(train_stats))



# bounding boxes
print(table(cleaned_train_set$label_id))


val_gt <- gt[tray_id %in% c(ARTVU_val_trays, PULDY_val_trays, ZEAMX_val_trays), ]
val_stats <- sapply(str_split(unique(val_gt$filename), "/"), `[`, 1)
print(table(val_stats))
print(table(val_gt$label_id))


test_gt <- gt[tray_id %in% c(ARTVU_test_trays, PULDY_test_trays, ZEAMX_test_trays), ]
test_stats <- sapply(str_split(unique(test_gt$filename), "/"), `[`, 1)
print(table(test_stats))
print(table(test_gt$label_id))


# statistics

# cleaned all trays
#rest set
val_test_set <- gt[tray_id %in% c(ARTVU_val_trays, ARTVU_test_trays, PULDY_val_trays, PULDY_test_trays, ZEAMX_val_trays, ZEAMX_test_trays), ]
cleaned_all_trays <- rbindlist(list(cleaned_train_set, val_test_set)) 


# overlapped plants within train_set
overlap_table <- cleaned_train_set %>%
  group_by(filename, tray_id) %>%
  nest() %>%
  mutate(overlapping = map(data, check_overlapping)) %>%
  unnest(overlapping)

overlap_table_no0 <- overlap_table[overlap_table$overlapping_count > 0, ] %>%
  .[c(1,2,4)] %>%
  as.data.frame() %>% 
  arrange(desc(overlapping_count))


# again for all trays
overlap_table <- cleaned_all_trays %>%
  group_by(filename, tray_id) %>%
  nest() %>%
  mutate(overlapping = map(data, check_overlapping)) %>%
  unnest(overlapping)


overlap_table_no0 <- overlap_table[overlap_table$overlapping_count > 0, ] %>%
  .[c(1,2,4)] %>%
  as.data.frame() %>% 
  arrange(desc(overlapping_count))


fwrite(overlap_table_no0, "/Users/mathias/Downloads/overlap_table_no0.csv")

# overlap per tray_id

overlap_tray <- overlap_table_no0 %>%
  group_by(tray_id) %>%
  summarize(total_overlaps = sum(overlapping_count))

# imgs with many overlapping plants
overlap_plants <- overlap_table %>%
  filter(overlapping_count != 0)


# imgs with many plants
multiple_instance <- cleaned_train_set[, .N, by=filename]


rm(total_c_undesired_trays, gt, undesired_trays)


uncleaned_train_dist <- table(vapply(strsplit(unique(train_set$filename),"/"), `[`, 1, FUN.VALUE=character(1)))
cleaned_train_dist <- table(vapply(strsplit(unique(cleaned_train_set$filename),"/"), `[`, 1, FUN.VALUE=character(1)))
print(uncleaned_train_dist)
print(cleaned_train_dist)


#

# random fold

calc_fold_size <- function(number, num_folds) {
  base_value <- floor(number / num_folds)
  # modulo rest
  remainder <- number %% num_folds
  result <- rep(base_value, num_folds)
  # rest verteilen
  if (remainder > 0) {
    result[1:remainder] <- result[1:remainder] + 1
  }
  return(result)
}

split_data_into_folds <- function(data, train_trays, num_folds) {
  files <- unique(data[tray_id %in% train_trays, ]$filename)
  frequencies <- calc_fold_size(length(files), num_folds)
  shuffle_folds <- sample(rep(1:num_folds, frequencies))
  return(data.table("filename" = files, "fold" = shuffle_folds))
}

apply_folds_to_dataset <- function(dataset, folds) {
  dataset_fold <- copy(dataset)
  dataset_fold$fold <- folds$fold[match(dataset_fold$filename, folds$filename)]
  return(dataset_fold)
}



folds <- 1:10
artvu_files <- unique(cleaned_train_set[tray_id %in% ARTVU_train_trays, ]$filename)
frequencies <- calc_fold_size(length(artvu_files), 10)
shuffle_folds <- sample(rep(folds, frequencies))

artvu_fold <- data.table("filename" = artvu_files, "fold" = shuffle_folds)


puldy_files <- unique(cleaned_train_set[tray_id %in% PULDY_train_trays, ]$filename)
frequencies <- calc_fold_size(length(puldy_files), 10)
shuffle_folds <- sample(rep(folds, frequencies))

puldy_fold <- data.table("filename" = puldy_files, "fold" = shuffle_folds)


zeamx_files <- unique(cleaned_train_set[tray_id %in% ZEAMX_train_trays, ]$filename)
frequencies <- calc_fold_size(length(zeamx_files), 10)
shuffle_folds <- sample(rep(folds, frequencies))

zeamx_fold <- data.table("filename" = zeamx_files, "fold" = shuffle_folds)

train_folds <- rbindlist(list(artvu_fold, puldy_fold, zeamx_fold))

cleaned_train_set_fold <- copy(cleaned_train_set)
cleaned_train_set_fold$fold <- train_folds$fold[match(cleaned_train_set_fold$filename, train_folds$filename)]
# proof
print(table(unique(cleaned_train_set_fold, by='filename')$fold))

## create json train
#cleaned_train_set_json <- cleaned_train_set_fold[fold %in% c(1,2,3,4),] %>%


cleaned_train_set_json <- cleaned_train_set_fold[fold==1, ] %>%
  mutate(category_id=match(.$label_id, desired_labels)) %>%
  mutate(width=rep(2454, dim(.)[1])) %>%
  mutate(height=rep(2056, dim(.)[1])) %>%
  mutate(image_id=match(.$filename, unique(.$filename))) # unique id for each image


fwrite(cleaned_train_set_json, "/Users/mathias/Code/PhentypingLoc/csvs/cleaned_train_set_v2_10.csv")

## create json validation

val_set <- gt[tray_id %in% c(ARTVU_val_trays, PULDY_val_trays, ZEAMX_val_trays), ] %>%
  .[!(filename %in% unique(undesired_entries$filename))]

val_set_json <- copy(val_set) %>%
  mutate(category_id=match(.$label_id, desired_labels)) %>%
  mutate(width=rep(2454, dim(.)[1])) %>%
  mutate(height=rep(2056, dim(.)[1])) %>%
  mutate(image_id=match(.$filename, unique(.$filename))) # unique id for each image

fwrite(val_set_json, "/Users/mathias/Code/PhentypingLoc/csvs/val_set.csv")



# test set with 4th weed class so stay uncleaned

# finish
test_set_json <- cleaned_train_set %>%
  mutate(category_id=match(.$label_id, desired_labels)) %>%
  mutate(width=rep(2454, dim(.)[1])) %>%
  mutate(height=rep(2056, dim(.)[1])) %>%
  mutate(image_id=match(.$filename, unique(.$filename))) # unique id for each image


fwrite(cleaned_train_set_json, "/Users/mathias/Code/PhentypingLoc/csvs/cleaned_train_set_v2.csv")


# create json for train + val

all_data <- gt[tray_id %in% c(ARTVU_train_trays, ARTVU_val_trays, PULDY_train_trays, PULDY_val_trays, ZEAMX_train_trays, ZEAMX_val_trays), ] %>%
  .[!(filename %in% unique(undesired_entries$filename))]


all_data_json <- copy(all_data) %>%
  mutate(category_id=match(.$label_id, desired_labels)) %>%
  mutate(width=rep(2454, dim(.)[1])) %>%
  mutate(height=rep(2056, dim(.)[1])) %>%
  mutate(image_id=match(.$filename, unique(.$filename))) # unique id for each image


fwrite(all_data_json, "/Users/mathias/Code/PhentypingLoc/csvs/real_data_v1_gt.csv")




# split 10% set into 2x 5%
cleaned_train_v2_10 <- fread("/Users/mathias/Code/PhentypingLoc/csvs/cleaned_train_set_v2_10.csv")
print(table(unique(cleaned_train_v2_10, by='filename')$label_id))
folds <- 2
artvu_fold <- split_data_into_folds(cleaned_train_v2_10, ARTVU_train_trays, folds)
puldy_fold <- split_data_into_folds(cleaned_train_v2_10, PULDY_train_trays, folds)
zeamx_fold <- split_data_into_folds(cleaned_train_v2_10, ZEAMX_train_trays, folds)

train_folds <- rbindlist(list(artvu_fold, puldy_fold, zeamx_fold))
cleaned_train_v2_5 <- apply_folds_to_dataset(cleaned_train_v2_10, train_folds)

# Proof
print(table(unique(cleaned_train_set_fold, by='filename')$fold))






## create json test

test_set <- gt[tray_id %in% c(ARTVU_test_trays, PULDY_test_trays, ZEAMX_test_trays), ] %>%
  .[!(filename %in% unique(undesired_entries$filename))]

test_set_json <- copy(test_set) %>%
  mutate(category_id=match(.$label_id, desired_labels)) %>%
  mutate(width=rep(2454, dim(.)[1])) %>%
  mutate(height=rep(2056, dim(.)[1])) %>%
  mutate(image_id=match(.$filename, unique(.$filename))) %>% # unique id for each image
  mutate(
    category_id = case_when(
      label_id == "POLAM" ~ 4,
      label_id == "Weed" ~ 5,
      TRUE ~ category_id
    )
  )
  
fwrite(test_set_json, "/Users/mathias/Code/PhentypingLoc/csvs/test_set.csv")
