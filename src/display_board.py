import numpy as np

BLACK           = "\033[0;30m"
DARK_GREY       = "\033[1;30m"
RED             = "\033[0;31m"
LIGHT_RED       = "\033[1;31m"
GREEN           = "\033[0;32m"
LIGHT_GREEN     = "\033[1;32m"
ORANGE          = "\033[0;33m"
YELLOW          = "\033[1;33m"
BLUE            = "\033[0;34m"
LIGHT_BLUE      = "\033[1;34m"
PURPLE          = "\033[0;35m"
LIGHT_PURPLE    = "\033[1;35m"
CYAN            = "\033[0;36m"
LIGHT_CYAN      = "\033[1;36m"
LIGHT_GREY      = "\033[0;37m"
WHITE           = "\033[1;37m"

COLOR_LIST = [WHITE  , YELLOW , LIGHT_BLUE , LIGHT_GREEN]
RESET_COLOR = WHITE
CLEAR_RIGHT = "\033[K"

'''

This class allow you to print beautiful array to visualize your data inside the UNIX terminal !!!
Here an example :

+--------------------------------------------------------------------------------+
|                 | class0            | class1            | class2               |
|                 | class3            | class4            | class5               |
|                 | class6            | class7            | class8               |
|                 | class9                                                       |
|================================================================================|
| precision       | 0.0               | 1.0               | 2.0                  |
|                 | 3.0               | 4.0               | 5.0                  |
|                 | 6.0               | 7.0               | 8.0                  |
|                 | 9.0                                                          |
|--------------------------------------------------------------------------------|
| recall          | 20.0              | 19.0              | 18.0                 |
|                 | 17.0              | 16.0              | 15.0                 |
|                 | 14.0              | 13.0              | 12.0                 |
|                 | 11.0                                                         |
+--------------------------------------------------------------------------------+

To get such result it's easy !!!!

Create an instance of DisplayBoard:

    b = DisplayBoard(["class{0}".format(i) for i in range(size)] , size_block = 20 , max_size = 60)

Add new rows:

    b.add_new_row("precision")
    b.add_new_row("recall")

Update the data inside each row:

    b.update_row_values("precision" , [k for k in range(size)])
    b.update_row_values("recall" , [20 - k for k in range(size)])

And now Display ! :

    b.display()

Care each data received by the DisplayBoard is copied !!!
'''

class DisplayBoard:
    def __init__(self , column_names , size_block = 20 , max_size = 60):
        """Create a new DisplayBoard
        Args:
            column_names: An array with the name of each column.
            size_block: The number of character allow inside one column
            max_size: The number of character allow for all the row (excluding the first row
                which contain the names of the row)
        """
        self.column_names = column_names
        self.row_names = []
        self.row_data = None
        self.block_size = size_block + 2 # +2 because we include 2 space one before and after the data
        self.size = self.block_size * (len(self.column_names) + 1)
        self.max_col = int(max_size / size_block)
        self.right_limit = min( (self.max_col + 1) * self.block_size , self.size)

    def add_new_row(self , row_name):
        """Add a new row to the board

        Args:
            row_name: The name of the new row
        """
        self.row_names.append(row_name)
        new_row = np.zeros( (len(self.column_names) , 1) )
        if self.row_data is None:
            self.row_data = new_row
        else:
            self.row_data = np.concatenate([self.row_data , new_row] , 1)

    def update_row_values(self , row_name , data):
        """Update one row data
        Args:
            row_name: The name of the updated row
            data: The new data to push inside the row
        """
        for i , name in enumerate(self.row_names):
            if name == row_name:
                self.row_data[ : , i ] = np.array(data)
                break

    def display(self):
        """Display the board
        """
        print(self.get_line(self.right_limit , "-" , "+") + '\n| ' , end='')

        self.display_line_of_values(self.column_names)
        print('{0}|{1}\n{2}'.format(self.get_str_move_to(self.right_limit + 2) , CLEAR_RIGHT , self.get_line(self.right_limit , "=" , "|")) )

        for y in range(len(self.row_names)):
            print('| {0} '.format(self.row_names[y]) , end='')
            values = self.row_data[: , y]
            self.display_line_of_values(values , ':.3f')

            print('{0} |'.format(self.get_str_move_to(self.right_limit + 1)))
            if y != len(self.row_names) - 1 :
                print(self.get_line(self.right_limit , "-" , "|"))

        print(self.get_line(self.right_limit , "-" , "+") , end='\n\n')


    def display_line_of_values(self , values , format = ''):
        for col , value in enumerate(values):
            sub_col = col % self.max_col

            if(col != 0 and sub_col == 0):
                print('{0} |{1}'.format(self.get_str_move_to(self.right_limit + 1) , CLEAR_RIGHT))
                print('|',end = ' ')

            str_row = ('{0}{1}{2}| ' + '{3' + format + '}' + '{4}').format(self.get_str_move_to( (sub_col + 1) * self.block_size - 1) ,
                                             CLEAR_RIGHT ,
                                             COLOR_LIST[(col // self.max_col) % len(COLOR_LIST)] ,
                                             value ,
                                             RESET_COLOR )
            print(str_row , end='')

    def get_line(self , size , content_char , border_char):
        str_line = border_char
        for i in range(size):
            str_line += content_char
        str_line += border_char

        return str_line

    def get_str_move_to(self , x_position):
        return "\033[10000;{0}H".format(x_position)



if __name__ == "__main__":
    #########################
    # EXAMPLE CODE
    #########################
    size = 10
    b = DisplayBoard(["class{0}".format(i) for i in range(size)] , size_block = 20 , max_size = 60)
    b.add_new_row("precision")
    b.add_new_row("recall")

    b.update_row_values("precision" , [k for k in range(size)])
    b.update_row_values("recall" , [20 - k for k in range(size)])
    b.display()
