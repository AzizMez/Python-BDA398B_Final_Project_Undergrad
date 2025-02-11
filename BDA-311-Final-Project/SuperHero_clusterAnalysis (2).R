##These two lines allow you to import the data, but you can also use the "Import Dataset" button in the Environment pane and the skip these lines.
music_data<-read.csv("Relation to Music Responses Processed_Augmented.csv")#run this line to load the selected file

##Packages for additional functions
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(dendextend) # for comparing two dendrograms
library(RColorBrewer) #For color scale in charts

###Data Preparation
# Prepare the data

clusterData <- scale(music_data) # scale the data
clusterData <- data.frame(clusterData) 

#Dissimilarity matrix -- this gives visibilty on the "distance" between super heros, blue means that they are similar red means they are different
d <- dist(clusterData, method = "euclidean")
fviz_dist(d, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

# Use the elbow method to determine the optimal number of clusters
fviz_nbclust(clusterData, kmeans, method = "wss") + geom_vline(xintercept = 4, linetype = 2)

###kMeans Clustering with 4 centers
k4 <- kmeans(clusterData, centers = 4, nstart = 100)
fviz_cluster(k4, data = clusterData)

sub_grp <- k4$cluster

####Code to visualize

# Add cluster variable to data -- this can be from any clustering method applied earlier
clusterData$Cluster <- sub_grp

# Visualize metrics relative to clusters
# This will summarize all variables for each cluster, including all columns
clustAverages <- clusterData %>%
  group_by(Cluster) %>%
  summarize(across(everything(), mean, na.rm = TRUE))

# Reshape the data for visualization
center_reshape <- clustAverages %>%
  pivot_longer(cols = -c(Cluster), names_to = "features", values_to = "values")

# Create a color palette for heatmaps
hm.palette <- colorRampPalette(rev(brewer.pal(11, 'RdYlBu')))(100)

# Generate a heatmap for visualizing cluster averages
ggplot(center_reshape, aes(x = features, y = Cluster, fill = values)) +
  geom_tile() +
  scale_fill_gradientn(colors = hm.palette) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Export Data with Cluster memberships for analysis in Tableau or Excel
music_data<-music_data[toKeep,]
music_data$Cluster<-sub_grp
write.csv(music_data, file = "music_data_cluster.csv")

