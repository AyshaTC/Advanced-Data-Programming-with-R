# Advanced Data Programming with R - Final Project
### Aysha Talakkal Cosumal (23200212)

This is a Readme file to briefly explain my app to you.

In my code, I've designed a shiny app to read and summarise the *Human Development Indicators* data.

The shiny app reads the following five countries from the website humdata.org for the purpose of this project:
1. Belgium
2. Mexico
3. Brazil
4. Ireland
5. Samoa

These countries are directly sourced from the website using the provided links, and therefore are not stored locally, and can be accessed anywhere.
I've also done some data cleaning in the beginning of the code, just so that my data is less messier. This includes:
* Removing the dummy header row from all datasets (this ensures that I can convert my columns to their appropriate data types instead of being read as 'character' bu default.
* Converting the 7th column, i.e, Value column to numeric type.
* Converting the rest of the columns to factor type, as they are all categorical (please note that the year column is a factor)

My app contains two panels: a summary tab and an Interactive App tab, where I can interact with my data by creating plots of my choosing and seeing rows of the data.

The app is so designed that I can choose multiple countries at the same time if I so choose, but it needs at least one country to perform analysis and plots.

The table only displays part of the data; the user can choose the number of rows and columns to be displayed.

The scatterplot helps visualise the data; the user can choose which field to use in the X-axis, Y-axis, and which field to color the plot by.
The histogram also has several choices for the user; the user can choose the number of bins, color of the histogram, choose to show density estimate, and output a graph description.

Additonal files can be uploaded in the bar in provided in the app (see "Upload Additional Dataset") but they have to be stored locally in the desktop using the app.

Thank you and enjoy using this app!
