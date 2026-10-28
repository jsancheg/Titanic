# =====================================================================
#  Who survives the Titanic?  Version with AGE AS A NUMBER (years)
#  Split  ->  Fit logistic regression  ->  Evaluate  ->  Predict
#
#  Needs individual-level ages, so it uses (in this order):
#    1. titanic::titanic_train   (Kaggle, 891 passengers)
#    2. carData::TitanicSurvival (1,309 passengers)
#  install.packages(c("shiny", "DT", "titanic"))   # or "carData"
#  Run with:  shiny::runApp("app_age_numeric.R")
# =====================================================================

library(shiny)
library(DT)

# ---------------------------------------------------------------------
# 1. DATA
#    Age must be a NUMBER here, so we need a passenger list with each
#    person's age in years. R's built-in datasets::Titanic is NOT used:
#    it only records Age as "Child" / "Adult".
# ---------------------------------------------------------------------
load_titanic <- function() {
  if (requireNamespace("titanic", quietly = TRUE)) {
    d <- titanic::titanic_train
    d <- d[, c("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")]
    d <- d[!is.na(d$Age) & d$Embarked != "", ]          # simple cleaning: complete cases
    d$Survived <- factor(d$Survived, levels = c(0, 1), labels = c("No", "Yes"))
    d$Pclass   <- factor(d$Pclass, levels = 1:3, labels = c("1st", "2nd", "3rd"))
    d$Sex      <- factor(d$Sex)
    d$Embarked <- factor(d$Embarked, levels = c("S", "C", "Q"),
                         labels = c("Southampton", "Cherbourg", "Queenstown"))
    d$Fare     <- round(d$Fare, 2)
    attr(d, "source") <- "Kaggle Titanic passenger list (titanic package), rows with missing Age/Embarked removed"
  } else if (requireNamespace("carData", quietly = TRUE)) {
    t <- carData::TitanicSurvival
    d <- data.frame(Survived = factor(t$survived, levels = c("no", "yes"), labels = c("No", "Yes")),
                    Pclass   = t$passengerClass,
                    Sex      = t$sex,
                    Age      = t$age)
    d <- d[!is.na(d$Age), ]                              # simple cleaning: complete cases
    attr(d, "source") <- "carData::TitanicSurvival passenger list, rows with missing Age removed"
  } else {
    stop("This version needs passengers' ages in years. Please run ",
         "install.packages(\"titanic\") (or install.packages(\"carData\")) and start the app again.")
  }
  d$Age <- as.numeric(d$Age)                             # Age is a number (years), not a factor
  src <- attr(d, "source")
  rownames(d) <- NULL
  d <- cbind(ID = seq_len(nrow(d)), d)
  attr(d, "source") <- src                               # cbind() drops custom attributes
  d
}

titanic_df <- load_titanic()
predictors <- setdiff(names(titanic_df), c("ID", "Survived"))

# Small helper for "pretty" percentages
pct <- function(x) sprintf("%.1f%%", 100 * x)

# ---------------------------------------------------------------------
# 2. USER INTERFACE
# ---------------------------------------------------------------------
ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { background:#f6f8fb; }
    .step { background:#fff; border-radius:10px; padding:14px 18px; margin-bottom:16px;
            box-shadow:0 1px 3px rgba(0,0,0,.08); }
    .step h4 { margin-top:0; color:#1f3b5b; }
    .stepnum { display:inline-block; background:#1f3b5b; color:#fff; border-radius:50%;
               width:26px; height:26px; text-align:center; line-height:26px; margin-right:6px; }
    .hint { color:#5a6b7d; font-size:0.92em; }
    .bigprob { font-size:2.4em; font-weight:700; color:#1f3b5b; }
    table.cm { border-collapse:collapse; margin:auto; font-size:1.1em; }
    table.cm td, table.cm th { border:1px solid #ccd5df; padding:10px 18px; text-align:center; }
    table.cm .ok  { background:#dff3e4; font-weight:700; }
    table.cm .bad { background:#fbe3e3; font-weight:700; }
  "))),
  withMathJax(),
  
  titlePanel("Who survives the Titanic? Logistic regression with Age in years"),
  p(class = "hint", "Data: ", attr(titanic_df, "source"), " | ",
    nrow(titanic_df), " people, outcome = Survived (Yes / No)."),
  
  # ---- STEP 1: SPLIT -------------------------------------------------
  div(class = "step",
      h4(span(class = "stepnum", "1"), "Split the data into training and test sets"),
      fluidRow(
        column(4,
               sliderInput("prop", "Proportion in the training set",
                           min = 0.5, max = 0.9, value = 0.7, step = 0.05),
               numericInput("seed", "Random seed (same seed = same split)", 2026, min = 1),
               actionButton("split", "Split the dataset", class = "btn-primary", icon = icon("scissors")),
               actionButton("reset", "Reset", icon = icon("rotate-left")),
               br(), br(),
               p(class = "hint", "Why split? The model learns from the training set. The test set
          plays the role of brand-new passengers it has never seen, so it gives an
          honest estimate of performance.")
        ),
        column(8, uiOutput("split_summary"))
      ),
      fluidRow(
        column(6, h5(textOutput("train_title")), DTOutput("train_tbl")),
        column(6, h5(textOutput("test_title")),  DTOutput("test_tbl"))
      )
  ),
  
  # ---- STEP 2: FIT + CONFUSION MATRIX --------------------------------
  div(class = "step",
      h4(span(class = "stepnum", "2"), "Fit a logistic regression and evaluate it on the test set"),
      fluidRow(
        column(5,
               actionButton("fit", "Fit the model", class = "btn-success", icon = icon("play")),
               br(), br(),
               helpText("$$\\log\\frac{p}{1-p}=\\beta_0+\\beta_1x_1+\\dots+\\beta_kx_k
                  \\quad\\Longleftrightarrow\\quad
                  p=\\frac{1}{1+e^{-(\\beta_0+\\beta_1x_1+\\dots+\\beta_kx_k)}}$$"),
               p(class = "hint", "\\(p\\) = probability of surviving. Each \\(\\beta\\) shifts the
          log-odds; \\(e^{\\beta}\\) is an odds ratio. A variable is called significant here if
          removing it makes the fit clearly worse (likelihood-ratio test, \\(p<0.05\\))."),
               sliderInput("thr", "Classification threshold: predict 'Yes' if p ≥",
                           min = 0.1, max = 0.9, value = 0.5, step = 0.05),
               h5("Which variables matter?"),
               tableOutput("sig_tbl")
        ),
        column(7,
               h5("Confusion matrix (test set)"),
               uiOutput("cm"),
               br(),
               uiOutput("metrics"),
               h5("Coefficients (full model, training set)"),
               tableOutput("coef_tbl"),
               uiOutput("age_note")
        )
      )
  ),
  
  # ---- STEP 3: PREDICT -----------------------------------------------
  div(class = "step",
      h4(span(class = "stepnum", "3"), "Predict survival for a new person"),
      fluidRow(
        column(5,
               p(class = "hint", "Only the significant variables appear below. The prediction uses a
          reduced model refitted on the training set with just these variables."),
               uiOutput("pred_inputs")
        ),
        column(7, uiOutput("pred_out"))
      )
  )
)

# ---------------------------------------------------------------------
# 3. SERVER
# ---------------------------------------------------------------------
server <- function(input, output, session) {
  
  rv <- reactiveValues(train = titanic_df, test = titanic_df[0, ],
                       split_done = FALSE, full = NULL, reduced = NULL, sig = NULL)
  
  # ---- Step 1: split ------------------------------------------------
  observeEvent(input$split, {
    set.seed(input$seed)
    n   <- nrow(titanic_df)
    idx <- sample(n, size = round(input$prop * n))
    rv$train <- titanic_df[sort(idx), ]
    rv$test  <- titanic_df[-idx, ]
    rv$split_done <- TRUE
    rv$full <- NULL; rv$reduced <- NULL; rv$sig <- NULL   # old model is now stale
  })
  
  observeEvent(input$reset, {
    rv$train <- titanic_df; rv$test <- titanic_df[0, ]
    rv$split_done <- FALSE; rv$full <- NULL; rv$reduced <- NULL; rv$sig <- NULL
  })
  
  output$train_title <- renderText(sprintf("Training set: %d people", nrow(rv$train)))
  output$test_title  <- renderText(sprintf("Test set: %d people", nrow(rv$test)))
  
  dt_opts <- list(pageLength = 8, scrollX = TRUE, dom = "tip")
  output$train_tbl <- renderDT(datatable(rv$train, rownames = FALSE, options = dt_opts))
  output$test_tbl  <- renderDT(datatable(rv$test,  rownames = FALSE, options = dt_opts))
  
  output$split_summary <- renderUI({
    surv <- function(d) if (nrow(d)) mean(d$Survived == "Yes") else NA
    if (!rv$split_done) {
      return(div(class = "hint", style = "padding-top:20px",
                 h5("Before splitting"), "All ", nrow(titanic_df), " people sit in the training table.
        Overall survival rate: ", strong(pct(surv(titanic_df))), ".", br(),
                 "Choose a proportion and press ", em("Split the dataset"), "."))
    }
    div(style = "padding-top:20px",
        h5("After splitting"),
        tags$ul(
          tags$li(sprintf("Training: %d people (%s) | survival rate %s",
                          nrow(rv$train), pct(nrow(rv$train) / nrow(titanic_df)), pct(surv(rv$train)))),
          tags$li(sprintf("Test: %d people (%s) | survival rate %s",
                          nrow(rv$test), pct(nrow(rv$test) / nrow(titanic_df)), pct(surv(rv$test))))
        ),
        p(class = "hint", "Question for you: are the two survival rates similar? Why does that matter?
        Try another seed and watch them move.")
    )
  })
  
  # ---- Step 2: fit --------------------------------------------------
  observeEvent(input$fit, {
    if (!rv$split_done) {
      showNotification("Split the dataset first (Step 1), so we keep a test set aside.",
                       type = "warning"); return()
    }
    f_full <- reformulate(predictors, response = "Survived")
    full   <- glm(f_full, data = rv$train, family = binomial)
    
    # Significance of each VARIABLE (factors tested as a whole) via likelihood-ratio test
    d1  <- drop1(full, test = "LRT")
    pv  <- d1[predictors, "Pr(>Chi)"]
    sig <- data.frame(Variable = predictors,
                      `p-value` = signif(pv, 3),
                      Significant = ifelse(pv < 0.05, "Yes", "No"),
                      check.names = FALSE)
    keep <- predictors[pv < 0.05]
    
    reduced <- if (length(keep))
      glm(reformulate(keep, response = "Survived"), data = rv$train, family = binomial) else
        glm(Survived ~ 1, data = rv$train, family = binomial)
    
    rv$full <- full; rv$reduced <- reduced; rv$sig <- sig
  })
  
  output$sig_tbl <- renderTable({
    req(rv$sig); rv$sig
  }, striped = TRUE)
  
  output$coef_tbl <- renderTable({
    req(rv$full)
    s <- summary(rv$full)$coefficients
    data.frame(Term = rownames(s),
               Estimate = round(s[, 1], 3),
               `Odds ratio` = round(exp(s[, 1]), 3),
               `p-value` = signif(s[, 4], 3),
               check.names = FALSE)
  }, striped = TRUE)
  
  # Age is numeric, so its coefficient is a SLOPE: the change in log-odds per extra year
  output$age_note <- renderUI({
    req(rv$full)
    b  <- coef(rv$full)[["Age"]]
    div(class = "hint",
        p(strong("Reading the Age row: "),
          sprintf("each extra year of age changes the log-odds by %.3f, so the odds are multiplied by
                e^(%.3f) = %.3f per year. Over 10 years: e^(10 \u00d7 %.3f) = %.3f, i.e. the odds of
                surviving change by %+.0f%% per decade.",
                  b, b, exp(b), b, exp(10 * b), 100 * (exp(10 * b) - 1))),
        p("Note the model assumes this effect is the same at every age (linear on the log-odds scale):
        going from 2 to 12 counts exactly as much as going from 50 to 60."))
  })
  
  # Predictions on the test set (full model)
  test_pred <- reactive({
    req(rv$full)
    p <- predict(rv$full, newdata = rv$test, type = "response")
    factor(ifelse(p >= input$thr, "Yes", "No"), levels = c("No", "Yes"))
  })
  
  output$cm <- renderUI({
    if (is.null(rv$full)) return(p(class = "hint", "Press ", em("Fit the model"), " to see results."))
    cm <- table(Predicted = test_pred(), Actual = rv$test$Survived)
    tags$table(class = "cm",
               tags$tr(tags$th(""), tags$th("Actual: No"), tags$th("Actual: Yes")),
               tags$tr(tags$th("Predicted: No"),
                       tags$td(class = "ok",  cm["No", "No"],  br(), tags$small("True negatives")),
                       tags$td(class = "bad", cm["No", "Yes"], br(), tags$small("False negatives"))),
               tags$tr(tags$th("Predicted: Yes"),
                       tags$td(class = "bad", cm["Yes", "No"],  br(), tags$small("False positives")),
                       tags$td(class = "ok",  cm["Yes", "Yes"], br(), tags$small("True positives")))
    )
  })
  
  output$metrics <- renderUI({
    req(rv$full)
    cm <- table(Predicted = test_pred(), Actual = rv$test$Survived)
    TN <- cm["No", "No"]; FN <- cm["No", "Yes"]; FP <- cm["Yes", "No"]; TP <- cm["Yes", "Yes"]
    acc  <- (TP + TN) / sum(cm)
    sens <- TP / (TP + FN)
    spec <- TN / (TN + FP)
    base <- max(mean(rv$test$Survived == "Yes"), mean(rv$test$Survived == "No"))
    tags$ul(
      tags$li(strong("Accuracy "), sprintf("(TP+TN)/N = %s", pct(acc))),
      tags$li(strong("Sensitivity "), sprintf("TP/(TP+FN) = %s  (survivors we correctly spot)", pct(sens))),
      tags$li(strong("Specificity "), sprintf("TN/(TN+FP) = %s  (non-survivors we correctly spot)", pct(spec))),
      tags$li(strong("Baseline "), sprintf("always guess the majority class = %s. Does the model beat it?", pct(base)))
    )
  })
  
  # ---- Step 3: predict ----------------------------------------------
  sig_vars <- reactive({ req(rv$sig); rv$sig$Variable[rv$sig$Significant == "Yes"] })
  
  output$pred_inputs <- renderUI({
    if (is.null(rv$sig)) return(p(class = "hint", "Fit the model first (Step 2)."))
    vars <- sig_vars()
    if (!length(vars)) return(p("No variable was significant in this training set; try another split."))
    lapply(vars, function(v) {
      x <- titanic_df[[v]]
      if (is.factor(x)) {
        selectInput(paste0("in_", v), v, choices = levels(x))
      } else {
        lab <- if (v == "Age") sprintf("Age in years (range %s to %s)", round(min(x), 2), max(x)) else
          sprintf("%s (range %s to %s)", v, min(x), max(x))
        numericInput(paste0("in_", v), lab, value = round(median(x), 1),
                     min = floor(min(x)), max = ceiling(max(x)), step = 1)
      }
    })
  })
  
  output$reduced_tbl <- renderTable({
    req(rv$reduced)
    s <- summary(rv$reduced)$coefficients
    data.frame(Term = rownames(s), Estimate = round(s[, 1], 3),
               `Odds ratio` = round(exp(s[, 1]), 3), check.names = FALSE)
  }, striped = TRUE)
  
  output$pred_out <- renderUI({
    req(rv$reduced, length(sig_vars()) > 0)
    vars <- sig_vars()
    vals <- lapply(vars, function(v) input[[paste0("in_", v)]])
    req(all(!vapply(vals, is.null, logical(1))), all(!is.na(unlist(vals))))
    
    nd <- titanic_df[1, vars, drop = FALSE]
    for (i in seq_along(vars)) {
      v <- vars[i]
      nd[[v]] <- if (is.factor(titanic_df[[v]]))
        factor(vals[[i]], levels = levels(titanic_df[[v]])) else as.numeric(vals[[i]])
    }
    eta <- predict(rv$reduced, newdata = nd, type = "link")
    p   <- 1 / (1 + exp(-eta))
    
    tagList(
      h5("Estimated probability of survival"),
      div(class = "bigprob", pct(p)),
      p(sprintf("Linear predictor (log-odds): η = %.3f  →  odds = e^η = %.3f  →  p = 1/(1+e^-η) = %.3f",
                eta, exp(eta), p)),
      p(strong(if (p >= input$thr) "Classified as: survives" else "Classified as: does not survive"),
        sprintf(" (threshold %.2f)", input$thr)),
      h5("Reduced model used for this prediction"),
      tableOutput("reduced_tbl"),
      p(class = "hint", "Try it: change one input at a time. Which change moves the probability most?
        Compare with the odds ratios above.")
    )
  })
}

shinyApp(ui, server)