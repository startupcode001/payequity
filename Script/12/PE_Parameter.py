import pandas as pd
import numpy as np

# Helper Functions Starts here #

def get_r2_option(r2):
    r2_format = round(r2*100,0)
    options = {
                  "series":[
                    {"max":100,
                     "min":0,
                     "splitNumber": 10,
                      "type":"gauge",
                      "axisLine":{
                        "lineStyle":{
                          "width":10,
                          "color":[
                            [
                              0.7,
                              "#9ca5af"
                            ],
                            [
                              1,
                              "#6bd47c"
                            ]
                          ]
                        }
                      },
                      "pointer":{
                        "itemStyle":{
                          "color":"auto"
                        }
                      },
                      "axisTick":{
                        "distance":-30,
                        "length":2,
                        "lineStyle":{
                          "color":"#fff",
                          "width":2
                        }
                      },
                      "splitLine":{
                        "distance":-5,
                        "length":10,
                        "lineStyle":{
                          "color":"#fff",
                          "width":1
                        }
                      },
                      "axisLabel":{
                        "color":"auto",
                        "distance":10,
                        "fontSize":10
                      },
                      "detail":{
                        "valueAnimation":True,
                        "formatter":"{value}%",
                        "color":"auto",
                        "fontSize": 15
                      },
                      "data":[
                        {
                          "value":r2_format
                        }
                      ]
                    }
                  ]
                }
    return options

def get_gender_gap_option(gap):
    gap_format = round(gap*100,0)
    options = {
                  "series":[
                    {"max":20,
                     "min":-20,
                     "splitNumber": 8,
                      "type":"gauge",
                      "axisLine":{
                        "lineStyle":{
                          "width":10,
                          "color":[
                              [
                              0.375,
                              "#9ca5af"
                            ],
                            [
                              1,
                              "#6bd47c"
                            ]
                          ]
                        }
                      },
                      "pointer":{
                        "itemStyle":{
                          "color":"auto"
                        }
                      },
                      "axisTick":{
                        "distance":-30,
                        "length":2,
                        "lineStyle":{
                          "color":"#fff",
                          "width":2
                        }
                      },
                      "splitLine":{
                        "distance":-5,
                        "length":10,
                        "lineStyle":{
                          "color":"#fff",
                          "width":1
                        }
                      },
                      "axisLabel":{
                        "color":"auto",
                        "distance":10,
                        "fontSize":10
                      },
                      "detail":{
                        "valueAnimation":True,
                        "formatter":"{value}%",
                        "color":"auto",
                        "fontSize": 15
                      },
                      "data":[
                        {
                          "value":gap_format
                        }
                      ]
                    }
                  ]
                }
    return options

