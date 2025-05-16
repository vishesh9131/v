"""
corerec employees mgm
[vishesh view]
1.initilly he will be a person object.
2.employee: inherit a class person details to ofc data.
3.employee: methods to assignwork and team for emp.
3.employee: methods to check leave taken.

[emp view]
4.
"""
class Person:
    def __init__(self,name,age,current_mood):
        self.name=name
        self.age=age
        self.current_mood=current_mood


class Employee(Person):
    team_avl=['cf_engine','uf_engine','devops','govn']
    
    def __init__(self,name,age,current_mood,team):
        super().__init__(name,age,current_mood)
        self.team=team
        self.applied_team=None
    
    def apply_for_team(self,choosen):
        if choosen in self.team_avl:
            self.applied_team=choosen
            print(f"applied for {choosen} sts: OK")
        else:
            print(f"{choosen}, not a team name")
        

    def team_info(self):
        for i in self.team_avl:
            print("-", i)
        print(f"your activity predicts the team : {self.applied_team}")


vish = Employee("vish",22,"g","cf_engine")
vish.apply_for_team("cf_engine")

vish.team_info()


