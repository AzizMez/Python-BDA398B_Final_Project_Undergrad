# Read the CSV file
music_data <- read.csv("C:/Users/User/Desktop/Relation to Music Responses.csv")

# Print the structure of the data
str(music_data)

# Print the updated data
print(music_data)

# Assuming your data frame is named 'music_data'

# Convert age into numerical categories
music_data$Age <- as.numeric(factor(music_data$Age, levels = c("16-Dec", "17-21", "22-26", "27-31", "32-36", "37-41", "42-46", "47-51", "52-56", "57-61", "62-66", "67-71", "72 and above")))

# Convert gender into numerical categories
music_data$Gender <- as.numeric(factor(music_data$Gender, levels = c("Male", "Female", "Non-binary", "Prefer not to say")))

# Convert education level into numerical categories
music_data$Education.level <- as.numeric(factor(music_data$Education.level, levels = c("School", "Bachelors", "Masters", "PhD")))

# Convert 'How.often.do.you.listen.to.music.' and 'How.important.is.music.in.your.life.' into numerical values
music_data$How.often.do.you.listen.to.music. <- as.numeric(music_data$How.often.do.you.listen.to.music.)
music_data$How.important.is.music.in.your.life <- as.numeric(music_data$How.important.is.music.in.your.life)

# Convert 'What.ways.do.you.engage.with.music.' into binary categories
ways_to_engage <- unique(unlist(strsplit(as.character(music_data$What.ways.do.you.engage.with.music.), ";")))
for (way in ways_to_engage) {
  music_data[paste("Engage_", gsub(" ", ".", way), sep = "")] <- as.integer(grepl(way, music_data$What.ways.do.you.engage.with.music.))
}

# Convert 'How.did.you.learn.music.' into numerical categories
music_data$How.did.you.learn.music. <- as.numeric(factor(music_data$How.did.you.learn.music., levels = c("I do not know anything about music only listen", "Private school", "Self learning", "Conservatory", "Internet", "Private tutoring")))
# Assuming your data frame is named 'music_data'

# ...

# Convert 'How.did.you.learn.music.' into numerical categories with correct factor levels
music_data$How.did.you.learn.music. <- as.numeric(factor(music_data$How.did.you.learn.music., levels = unique(music_data$How.did.you.learn.music.)))

# ...

# Display the updated data frame
View(music_data)

# View the modified dataframe
View(music_data)

# ...

# Convert 'How.did.you.learn.music.' into numerical categories with correct factor levels
music_data$How.did.you.learn.music. <- as.numeric(factor(music_data$How.did.you.learn.music., levels = unique(music_data$How.did.you.learn.music.)))

# Replace NAs in "How.did.you.learn.music." with a numerical category (e.g., 0)
music_data$How.did.you.learn.music. <- ifelse(is.na(music_data$How.did.you.learn.music.), 0, music_data$How.did.you.learn.music.)

# Convert the column to numeric
music_data$How.did.you.learn.music. <- as.numeric(music_data$How.did.you.learn.music.)

# Display the updated data frame
View(music_data)

write.csv(music_data, file = "Relation to Music Responses Processed.csv", row.names = FALSE)


