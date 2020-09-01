# Feature_engineering - approach description

### 1. Missing values 

For missing values I have decided to inject median value for numeric types and most frequent one for category types. 
For training dataset there is record missing 'Embarked'. There are multiple records missing 'Age' value, dropping them would result in huge loose. 
Also majority of the records was missing 'Cabin' value, that why I also dropped this column. 

### 2. Feature split

Something what I wanted to do was to decompose 'Ticket' variable. It had multiple unique variables, however some of the tickets contains prefix.
I'm not fully aware of the origin and meaning of this prefix, yet I decided to split Ticket to two columns - 'Ticket_prefix' and 'Ticket_number'.
For some ticket prefixes there is a significant correlation with target feature. Same prefix can indicate relation between people. 
I decided to keep ticket number as it can indicate location on the ship. What I noticed is that majority of 3rd class tickets has higher ticket number.
Other approach with handling prefix would be to only keep bool information if ticket has prefix or not. 'Ticket' feature without decomposition has more than 600 unique categories.

### 3. Dropping features

I also decided to drop 'Name', 'Cabin' and 'Ticket'(but keeping its decomposed values - 'Ticket_prefix' and 'Ticket_number').

### 4. Dummification - OneHotEncoder

Instead of changing 'Sex' and 'Embarked' to numeric, I have decided to dummify it. Changing values for 'Embarked' would imply an order.
I also used OneHotEncoder on 'Ticket_prefix' and that contributed to 42 new features(categories). Having no prefix on the ticket is also a category.

### 5. Normalization

To unify a domain I have used min_max scaler. Improvement would include removal of the outliers before applying normalization. 

### 6. Building features 

I decided to keep 'FamilySize'. However, I removed 'IsAlone', because I think that is a redundant feature. 
'FamilySize' equal one provides this information, so I decided that keeping 'IsAlone' is not of high value. 

### 7. Feature aggregation 

For this dataset I did not notice a need to perform feature aggregation. 
